// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! GPU partition mode enums. See `../wiki/concepts/gpu-partition-modes.md`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum ComputePartitionMode {
    Spx,
    Dpx,
    Qpx,
    Cpx,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum MemoryPartitionMode {
    Nps1,
    Nps2,
    Nps4,
    #[default]
    Unknown,
}
