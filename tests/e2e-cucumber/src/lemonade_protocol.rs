// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! HTTP protocol fixture, not a real Lemonade runtime or inference engine.
//! Runs on its own thread so blocking CLI subprocesses cannot stall its routes.

use std::sync::{Arc, Mutex};

use axum::{Json, Router, extract::State, routing::get};
use serde_json::{Value, json};

use crate::http_server::{self, ServerHandle};

#[derive(Debug)]
pub struct LemonadeProtocolServer {
    server: ServerHandle,
    loaded: Arc<Mutex<Vec<String>>>,
}

impl LemonadeProtocolServer {
    pub fn start(models: &[&str]) -> Self {
        let loaded = Arc::new(Mutex::new(models.iter().map(|s| (*s).to_owned()).collect()));
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
            .with_state(Arc::clone(&loaded));
        Self {
            server: http_server::spawn_on_own_thread(app),
            loaded,
        }
    }

    pub const fn port(&self) -> u16 {
        self.server.port()
    }

    pub fn base_url(&self) -> String {
        self.server.base_url()
    }

    pub fn set_loaded(&self, models: &[&str]) {
        *self.loaded.lock().expect("fixture state poisoned") =
            models.iter().map(|s| (*s).to_owned()).collect();
    }
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
