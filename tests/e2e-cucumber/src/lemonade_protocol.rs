// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! HTTP protocol fixture, not a real Lemonade runtime or inference engine.
//! Runs on its own thread so blocking CLI subprocesses cannot stall its routes.

use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    extract::{Request, State},
    middleware::{self, Next},
    response::Response,
    routing::get,
};
use serde_json::{Value, json};

use crate::http_server::{self, ServerHandle};

#[derive(Debug)]
pub struct LemonadeProtocolServer {
    server: ServerHandle,
    loaded: Arc<Mutex<Vec<String>>>,
    stray: Arc<Mutex<Vec<String>>>,
}

impl LemonadeProtocolServer {
    pub fn start(models: &[&str]) -> Self {
        let loaded = Arc::new(Mutex::new(models.iter().map(|s| (*s).to_owned()).collect()));
        let stray = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/api/v1/health", get(health))
            .route(
                "/api/v1/stats",
                get(|| async {
                    Json(json!({
                        "tokens_per_second": 37.5,
                        "time_to_first_token": 0.05,
                        "input_tokens": 8,
                        "output_tokens": 16
                    }))
                }),
            )
            .with_state(Arc::clone(&loaded))
            .layer(middleware::from_fn_with_state(
                Arc::clone(&stray),
                record_stray,
            ));
        Self {
            server: http_server::spawn_on_own_thread(app),
            loaded,
            stray,
        }
    }

    pub const fn port(&self) -> u16 {
        self.server.port()
    }

    pub fn base_url(&self) -> String {
        self.server.base_url()
    }

    /// Every request that was not a `GET` of the health or stats route, as
    /// `"<METHOD> <path>"`. A client that unloads, loads, or stops a model on
    /// this server, or probes any other route, shows up here.
    pub fn stray_requests(&self) -> Vec<String> {
        self.stray.lock().expect("fixture state poisoned").clone()
    }

    pub fn set_loaded(&self, models: &[&str]) {
        *self.loaded.lock().expect("fixture state poisoned") =
            models.iter().map(|s| (*s).to_owned()).collect();
    }
}

async fn record_stray(
    State(stray): State<Arc<Mutex<Vec<String>>>>,
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let allowed = method == axum::http::Method::GET
        && matches!(path.as_str(), "/api/v1/health" | "/api/v1/stats");
    if !allowed {
        stray
            .lock()
            .expect("fixture state poisoned")
            .push(format!("{method} {path}"));
    }
    next.run(request).await
}

async fn health(State(loaded): State<Arc<Mutex<Vec<String>>>>) -> Json<Value> {
    let loaded = loaded.lock().expect("fixture state poisoned");
    Json(json!({
        "status": "ok",
        "model_loaded": loaded.first(),
        "all_models_loaded": loaded
            .iter()
            .map(|model| json!({"model_name": model}))
            .collect::<Vec<_>>()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn records_requests_other_than_get_health_and_stats() {
        let fixture = LemonadeProtocolServer::start(&["a", "b"]);
        let base = fixture.base_url();
        let client = reqwest::Client::new();
        client
            .get(format!("{base}/api/v1/health"))
            .send()
            .await
            .unwrap();
        client
            .get(format!("{base}/api/v1/stats"))
            .send()
            .await
            .unwrap();
        assert!(fixture.stray_requests().is_empty());

        client
            .post(format!("{base}/api/v1/unload"))
            .send()
            .await
            .unwrap();
        client
            .get(format!("{base}/api/v1/models"))
            .send()
            .await
            .unwrap();
        client
            .delete(format!("{base}/api/v1/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(
            fixture.stray_requests(),
            vec![
                "POST /api/v1/unload",
                "GET /api/v1/models",
                "DELETE /api/v1/health"
            ]
        );
    }
}
