// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// Cucumber step functions share one uniform signature — `async fn(world: &mut
// E2eWorld, ...)` — so the `#[given/when/then]` macros can register them the
// same way. Many steps neither `.await` nor mutate the world; allow both rather
// than splitting the step API into sync/async and mut/non-mut variants.
#![allow(clippy::unused_async, clippy::needless_pass_by_ref_mut)]

use std::path::PathBuf;

use cucumber::{World as _, WriterExt as _};
use e2e_cucumber::mock_server::MockServer;
use tempfile::TempDir;

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
    /// Name of the scenario currently executing, set by the `before` hook. Used
    /// to tie each recorded `rocm` invocation to its scenario so the coverage
    /// report can join commands to pass/fail results.
    pub current_scenario: Option<String>,
    /// Per-scenario isolated config/data/cache root. A `TempDir` so it is unique
    /// per World and auto-removed on drop; using `tempfile` also keeps the OS
    /// temp-dir lookup out of our source (avoids a CodeQL path-injection
    /// false positive on `env::temp_dir()`).
    pub isolated_root: Option<TempDir>,
}

/// A persistent directory shared across scenarios for heavy, immutable artifacts
/// (TheRock runtime wheels, HF model weights, engine venvs). Set by CI to a path
/// on the runner's persistent disk; unset for local runs, where every scenario
/// stays fully isolated (nothing shared).
///
/// Sharing these read-only artifacts avoids re-downloading multi-GB runtimes and
/// model weights per scenario. Only immutable artifacts are shared — service
/// records, config, and per-service engine state stay isolated per scenario.
fn shared_cache_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("E2E_SHARED_CACHE_DIR")?;
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

impl Default for E2eWorld {
    fn default() -> Self {
        // A fresh TempDir per World gives each scenario its own isolated
        // config/data/cache root (unique — concurrent scenarios never share a
        // tree) and auto-removes it on drop.
        let root = TempDir::with_prefix("rocm-e2e-").expect("failed to create temp dir");
        for sub in ["config", "data", "cache"] {
            std::fs::create_dir_all(root.path().join(sub)).ok();
        }

        // Share the immutable TheRock runtimes across scenarios: point this
        // scenario's <data>/runtimes at the shared dir via a symlink, so the
        // ~3.3GB wheel + its registry manifest are downloaded once per runner,
        // not once per scenario. Write-once artifacts, safe to share; service
        // state lives elsewhere under <data> and stays isolated.
        if let Some(shared) = shared_cache_dir() {
            let shared_runtimes = shared.join("runtimes");
            if std::fs::create_dir_all(&shared_runtimes).is_ok() {
                let link = root.path().join("data").join("runtimes");
                #[cfg(unix)]
                let _ = std::os::unix::fs::symlink(&shared_runtimes, &link);
                #[cfg(windows)]
                let _ = std::os::windows::fs::symlink_dir(&shared_runtimes, &link);
            }
        }

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
            current_scenario: None,
            isolated_root: Some(root),
        }
    }
}

impl E2eWorld {
    pub fn isolate_cmd(&self, cmd: &mut std::process::Command) {
        if let Some(root) = &self.isolated_root {
            let root = root.path();
            cmd.env("ROCM_CLI_CONFIG_DIR", root.join("config"));
            cmd.env("ROCM_CLI_DATA_DIR", root.join("data"));
            cmd.env("ROCM_CLI_CACHE_DIR", root.join("cache"));
        }
        // Share the immutable heavy caches across scenarios when CI provides a
        // persistent shared dir (see shared_cache_dir): HF model weights via
        // HF_HOME (engines honour it for both download and discovery; weights are
        // content-addressed and immutable) and the vLLM engine venv via
        // ROCM_CLI_ENGINE_ENVS_ROOT (redirects only the envs/ leaf, so per-service
        // engine state stays isolated). The runtimes dir is shared via a symlink
        // set up in `default()`.
        if let Some(shared) = shared_cache_dir() {
            cmd.env("HF_HOME", shared.join("huggingface"));
            cmd.env("ROCM_CLI_ENGINE_ENVS_ROOT", shared.join("engine-envs"));
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
        let services = root.path().join("data").join("services");
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
        // A scenario that ran `rocm serve --managed` left a DETACHED supervisor +
        // engine process (vLLM / llama-server) that outlives this harness — the
        // TempDir drop below removes the on-disk record but never kills those
        // processes, so on a persistent runner they accumulate and hold the GPU.
        // Stop every managed service recorded in this scenario's isolated root
        // before the directory is removed. Best-effort: this is teardown, so any
        // failure is ignored rather than panicking (which would abort the run).
        if let Some(root) = &self.isolated_root {
            stop_managed_services(root.path());
        }
        // `isolated_root` is a `TempDir`; its own Drop removes the directory.
    }
}

/// Stop every ROCm-managed service recorded under an isolated root's
/// `data/services/*.json`, so detached engine processes don't leak past the
/// scenario. Black-box: reads the service_id from each on-disk record and calls
/// `rocm services stop <id> --yes` with the same isolated env the scenario used.
fn stop_managed_services(root: &std::path::Path) {
    let services_dir = root.join("data").join("services");
    let Ok(entries) = std::fs::read_dir(&services_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Ok(record) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        let Some(service_id) = record.get("service_id").and_then(|v| v.as_str()) else {
            continue;
        };
        // The planted mock record has no real process to stop; skip it.
        if service_id == "e2e-mock" {
            continue;
        }
        let mut cmd = std::process::Command::new(rocm_binary());
        cmd.args(["services", "stop", service_id, "--yes"]);
        cmd.env("ROCM_CLI_CONFIG_DIR", root.join("config"));
        cmd.env("ROCM_CLI_DATA_DIR", root.join("data"));
        cmd.env("ROCM_CLI_CACHE_DIR", root.join("cache"));
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());
        let _ = cmd.status();
    }
}

// ── Shared helpers ─────────────────────────────────────────────────

pub fn rocm_binary() -> String {
    std::env::var("ROCM_CLI_BINARY").unwrap_or_else(|_| "rocm".to_string())
}

/// Spawn the real `rocm` binary with the scenario's isolated env, returning
/// `(stdout, stderr, rc)`. Every scenario goes through here, so this is also
/// where each invocation is recorded for the command-coverage report.
pub fn run_rocm(world: &E2eWorld, args: &[&str]) -> (String, String, i32) {
    let binary = rocm_binary();
    let mut cmd = std::process::Command::new(&binary);
    cmd.args(args);
    world.isolate_cmd(&mut cmd);
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to run {binary}: {e}"));
    let rc = output.status.code().unwrap_or(-1);
    record_command(world.current_scenario.as_deref(), args, rc);
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
        rc,
    )
}

/// Append one `rocm` invocation to `results/commands.jsonl` so the consolidated
/// report can build a command × platform coverage table tied to real results.
/// Best-effort: a recording failure must never fail a scenario.
fn record_command(scenario: Option<&str>, args: &[&str], rc: i32) {
    let subcommand = derive_subcommand(args);
    let engine = flag_value(args, "--engine");
    let model = positional_model(args);
    let record = serde_json::json!({
        "scenario": scenario,
        "argv": args,
        "rc": rc,
        "subcommand": subcommand,
        "model": model,
        "engine": engine,
    });
    let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/results"));
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(mut line) = serde_json::to_string(&record) {
        line.push('\n');
        // Append; concurrent scenarios each add their own lines.
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("commands.jsonl"))
        {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// The signature used to group invocations in the coverage table: the leading
/// subcommand plus the flags that materially change behaviour (e.g. `--engine`,
/// `--managed`), but not values like the model name (shown in its own column).
fn derive_subcommand(args: &[&str]) -> String {
    // First non-flag token(s): most rocm subcommands are one word (`serve`,
    // `examine`, `chat`), a few are two (`install sdk`, `services list`,
    // `runtimes activate`).
    let words: Vec<&str> = args
        .iter()
        .take_while(|a| !a.starts_with('-'))
        .copied()
        .collect();
    let base = match words.as_slice() {
        [] => "rocm".to_string(),
        [one] => (*one).to_string(),
        [first, second, ..] => format!("{first} {second}"),
    };
    // Note the behaviour-shaping flags so `serve` vs `serve --engine vllm` vs
    // `serve` (default engine) are distinct rows.
    let mut sig = format!("rocm {base}");
    if args.contains(&"--engine") {
        sig.push_str(" --engine");
    } else if base == "serve" {
        sig.push_str(" (default engine)");
    }
    sig
}

/// Value following `flag` in argv, if present (e.g. the engine after `--engine`).
fn flag_value(args: &[&str], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| *a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| (*s).to_string())
}

/// The model positional for model-taking subcommands (`serve <model>`). Returns
/// the first non-flag token after the subcommand that looks like a model ref.
fn positional_model(args: &[&str]) -> Option<String> {
    // Only `serve` takes a model positional in this suite.
    if args.first() != Some(&"serve") {
        return None;
    }
    args.iter()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .map(|s| (*s).to_string())
}

/// How long a single inference request may take before the harness gives up.
///
/// This bounds the test's wall-clock, not the product: a genuinely hung backend
/// (e.g. EAI-7052, lemonade falling back to Vulkan) would otherwise block the
/// HTTP call forever and let a known-bug scenario run until the CI job limit.
/// Capping it turns the hang into a prompt failure — exactly the expected
/// outcome for an `@expected-failure` scenario. 10s is ample for a small model
/// that is already loaded (serve readiness is waited for separately) to answer a
/// one-word prompt; override with `E2E_INFERENCE_TIMEOUT_SECS` if needed.
fn inference_timeout_secs() -> u64 {
    std::env::var("E2E_INFERENCE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10)
}

pub async fn send_chat(world: &mut E2eWorld) {
    let endpoint = world.endpoint.as_ref().expect("no endpoint configured");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(inference_timeout_secs()))
        .build()
        .expect("failed to build HTTP client");

    let models_url = format!("{endpoint}/models");
    let resp: serde_json::Value = client
        .get(&models_url)
        .send()
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
        // Record the scenario name on the World before each scenario so every
        // `rocm` invocation can be tied back to its scenario for the coverage
        // report.
        .before(|_feature, _rule, scenario, world| {
            world.current_scenario = Some(scenario.name.clone());
            Box::pin(async {})
        })
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

    // Known-bugs mode (`cargo xtask e2e --expect-failures`): the suite is filtered
    // to `@expected-failure` scenarios, whose failing IS the expected outcome. A
    // parse/hook error still fails outright (the run didn't execute cleanly);
    // otherwise invert the step-failure signal via xfail/XPASS accounting.
    if std::env::var_os("E2E_EXPECT_FAILURES").is_some() {
        if summary.parsing_errors() > 0 || summary.hook_errors() > 0 {
            eprintln!(
                "E2E run errored: {} parsing error(s), {} hook error(s)",
                summary.parsing_errors(),
                summary.hook_errors(),
            );
            std::process::exit(1);
        }
        let xfail = e2e_cucumber::report::evaluate_xfail(&dir.join("report.json"))
            .expect("failed to evaluate xfail report");
        eprintln!(
            "Known-bugs run: {} scenario(s) failed as expected (xfail).",
            xfail.xfail,
        );
        if !xfail.is_ok() {
            for name in &xfail.xpass {
                eprintln!(
                    "XPASS: '{name}' is tagged @expected-failure but PASSED \u{2014} the bug \
                     appears fixed; remove the tag so it joins the blocking suite.",
                );
            }
            for name in &xfail.untagged_failures {
                eprintln!(
                    "FAIL: '{name}' failed but is not tagged @expected-failure \u{2014} a real \
                     regression in the known-bugs run.",
                );
            }
            std::process::exit(1);
        }
        return;
    }

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
