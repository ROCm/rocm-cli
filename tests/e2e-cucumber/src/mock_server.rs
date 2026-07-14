// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::net::SocketAddr;

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
    state: ServerState,
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
            .with_state(state.clone());

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

        Self {
            addr,
            state,
            shutdown: tx,
        }
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

async fn handle_models(State(state): State<ServerState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": state.model_name, "object": "model"}]
    }))
}

async fn handle_chat(State(state): State<ServerState>, Json(body): Json<Value>) -> Json<Value> {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("<missing>")
        .to_string();

    Json(json!({
        "id": "mock-completion-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a mock response for testing."
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
