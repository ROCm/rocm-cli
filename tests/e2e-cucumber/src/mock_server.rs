// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::net::SocketAddr;
use std::path::Path;

use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::{get, post};
use serde_json::{Value, json};
use tokio::net::TcpListener;

#[derive(Clone)]
struct ServerState {
    model_name: String,
}

pub struct MockServer {
    addr: SocketAddr,
    shutdown: tokio::sync::oneshot::Sender<()>,
}

impl std::fmt::Debug for MockServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockServer")
            .field("addr", &self.addr)
            .finish_non_exhaustive()
    }
}

impl MockServer {
    pub async fn start(model_name: &str) -> Self {
        let state = ServerState {
            model_name: model_name.to_string(),
        };

        let app = Router::new()
            .route("/v1/models", get(handle_models))
            .route("/models", get(handle_models))
            .route("/v1/chat/completions", post(handle_chat))
            .route("/chat/completions", post(handle_chat))
            .with_state(state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();

        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    rx.await.ok();
                })
                .await
                .ok();
        });

        Self { addr, shutdown: tx }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    pub const fn port(&self) -> u16 {
        self.addr.port()
    }

    pub fn stop(self) {
        self.shutdown.send(()).ok();
    }
}

/// Write a managed-service record pointing the CLI at a mock server on `port`.
///
/// Drops the JSON into `services_dir` (`<data>/services/`) exactly as `rocm serve
/// --managed` would. Shared by the cucumber `World` and the standalone
/// `rocm-demo-env` binary so the on-disk schema lives in one place. Black-box:
/// plain JSON matching the CLI's on-disk schema, not a typed import from the
/// rocm-cli crates.
pub fn write_service_record(services_dir: &Path, model: &str, port: u16) {
    std::fs::create_dir_all(services_dir).expect("failed to create services dir");

    // The CLI only extracts host:port from `endpoint_url` and appends
    // `/v1/models` for its readiness probe, which the mock serves.
    let record = json!({
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
        "manifest_path": services_dir.join("e2e-mock.json"),
        "log_path": services_dir.join("e2e-mock.log"),
        "engine_state_path": services_dir.join("e2e-mock.state.json"),
        "created_at_unix_ms": 1_700_000_000_000_u64,
    });
    std::fs::write(
        services_dir.join("e2e-mock.json"),
        serde_json::to_vec_pretty(&record).expect("failed to serialize record"),
    )
    .expect("failed to write service record");
}

async fn handle_models(State(state): State<ServerState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": state.model_name, "object": "model"}]
    }))
}

async fn handle_chat(Json(body): Json<Value>) -> Json<Value> {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("<missing>")
        .to_string();

    // Tests assert on the request contract, not the reply text, so the default
    // is a fixed stub. Demos (`rocm-demo-env`) set `ROCM_MOCK_CHAT_REPLY` to a
    // realistic answer so the recorded screencast reads naturally.
    let content = std::env::var("ROCM_MOCK_CHAT_REPLY")
        .unwrap_or_else(|_| "This is a mock response for testing.".to_string());

    Json(json!({
        "id": "mock-completion-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }))
}
