// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! /proc scan for non-container deployments (Strix Halo). Stub.

use rocm_dash_core::traits::{CollectorError, DiscoveredService, Result, ServiceDiscovery};

#[derive(Debug, Default)]
pub struct ProcDiscovery;

impl ProcDiscovery {
    pub const fn new() -> Self {
        Self
    }
}

impl ServiceDiscovery for ProcDiscovery {
    fn name(&self) -> &'static str {
        "proc"
    }

    fn discover(&self) -> Result<Vec<DiscoveredService>> {
        // TODO: walk /proc, match comm == "llama-server" | "lemonade-server".
        Err(CollectorError::Unsupported("proc discovery stub".into()))
    }
}
