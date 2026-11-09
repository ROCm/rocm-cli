// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Fixtures shared by the tests of more than one launch submodule.

use anyhow::Result;
use rocm_core::{AppPaths, ManagedServiceRecord};

/// A record with every reuse-relevant field set to the "reusable" value, so
/// each test below breaks exactly one term and nothing else.
pub(super) fn reusable_record(paths: &AppPaths) -> ManagedServiceRecord {
    let mut record = ManagedServiceRecord::new(
        paths,
        "vllm-qwen-1000",
        "vllm",
        "qwen",
        "qwen-canonical".to_owned(),
        "127.0.0.1",
        8000,
        "managed",
        0,
        None,
        None,
        Some("gpu_required".to_owned()),
    );
    record.engine_recipe_json = None;
    record.requires_api_key = true;
    record
}

pub(super) fn write_live_service_record(
    paths: &AppPaths,
    service_id: &str,
    canonical_model_id: &str,
    port: u16,
    status: &str,
) -> Result<ManagedServiceRecord> {
    let mut record = ManagedServiceRecord::new(
        paths,
        service_id,
        "lemonade",
        canonical_model_id,
        canonical_model_id.to_owned(),
        rocm_core::DEFAULT_LOCAL_HOST,
        port,
        "managed",
        std::process::id(),
        None,
        None,
        None,
    );
    record.status = status.to_owned();
    record.engine_pid = Some(std::process::id());
    record.write()?;
    Ok(record)
}
