// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! How a request is turned into the artifact an install fetches. Only module
//! declarations and re-exports live here, no logic: `tarball` resolves the SDK
//! tarball from the catalog's index, `wheel` reads what the wheel indexes
//! publish, and `therock.rs` reads the names below.

mod tarball;
mod wheel;

pub(super) use self::tarball::{resolve_tarball_artifact, resolve_tarball_artifact_with_timeout};
pub(super) use self::wheel::{
    AggregateDeviceTarget, ResolvedAggregateWheelSource, load_simple_index_versions,
    package_release_key, package_rocm_suffix, parse_aggregate_device_targets,
    validate_aggregate_index_layout,
};
