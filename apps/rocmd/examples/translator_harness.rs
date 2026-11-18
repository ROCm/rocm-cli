// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! SPIKE ONLY — not for commit. Runs the real Anthropic translation front over
//! an already-running OpenAI-compatible engine and registers it as a ready
//! managed service, so `rocm agents run --local` can be tried without
//! installing an engine.
//!
//! usage: translator_harness <engine-port> <service-id>   (ROCM_CLI_DATA_DIR set)

use std::time::Duration;

use rocm_core::{AppPaths, ManagedServiceRecord};
use rocmd::anthropic_proxy::{AnthropicProxyConfig, spawn_alongside_engine};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let port: u16 = args[1].parse()?;
    let service_id = &args[2];
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    let state_path = paths
        .data_dir
        .join(format!("{service_id}.engine-state.json"));
    std::fs::write(&state_path, "{}")?;
    spawn_alongside_engine(AnthropicProxyConfig {
        host: "127.0.0.1".into(),
        engine_port: port,
        served_model: "mock-local-model".into(),
        state_path: state_path.clone(),
        probe_api_key: None,
    });
    let url =
        rocm_core::wait_for_published_anthropic_endpoint(&state_path, Duration::from_secs(30))
            .ok_or_else(|| anyhow::anyhow!("translator did not publish an endpoint"))?;
    let mut record = ManagedServiceRecord::new(
        &paths,
        service_id.as_str(),
        "lemonade",
        "mock-local-model",
        "mock-local-model",
        "127.0.0.1",
        port,
        "managed",
        std::process::id(),
        None,
        None,
        None,
    );
    "ready".clone_into(&mut record.status);
    record.anthropic_endpoint_url = Some(url.clone());
    record.write()?;
    println!("READY {service_id} anthropic={url}");
    loop {
        std::thread::sleep(Duration::from_hours(1));
    }
}
