// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! llama-server /slots collector. Stub.

use rocm_dash_core::traits::{
    CollectorError, DiscoveredService, InstanceMetrics, InstanceSample, Result,
};

#[derive(Debug, Default)]
pub struct LlamaSlotsCollector;

impl LlamaSlotsCollector {
    pub const fn new() -> Self {
        Self
    }
}

impl InstanceMetrics for LlamaSlotsCollector {
    fn name(&self) -> &'static str {
        "llama-slots"
    }

    fn fetch(&self, _svc: &DiscoveredService) -> Result<InstanceSample> {
        // TODO: GET http://localhost:{port}/slots; count non-idle entries.
        // llama.cpp has no waiting queue → set waiting_reqs = Some(0).
        Err(CollectorError::Unsupported("llama-slots stub".into()))
    }
}
