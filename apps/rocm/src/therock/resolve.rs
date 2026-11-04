// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! How a request is turned into the artifact an install fetches. Only module
//! declarations and re-exports live here, no logic: `tarball` resolves the SDK
//! tarball from the catalog's index, `wheel` resolves the wheel stack from the
//! wheel indexes, and the rest of `therock` reads the names below.

mod tarball;
mod wheel;

pub(super) use self::tarball::{resolve_tarball_artifact, resolve_tarball_artifact_with_timeout};
#[cfg(test)]
pub(super) use self::wheel::select_rocm_version;
pub(super) use self::wheel::{
    AggregateDeviceTarget, canonical_auto_select_needs_next_python, resolve_pip_runtime,
    resolve_pip_runtime_with_timeout, unverified_torchaudio_note, wheel_runtime_composition,
};
