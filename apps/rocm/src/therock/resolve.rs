// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! How a request is turned into the artifact an install fetches. Only module
//! declarations and re-exports live here, no logic: `tarball` resolves the SDK
//! tarball from the catalog's index, `wheel` resolves the wheel stack from the
//! wheel indexes (with `uv_access` deciding who may fetch `uv` while it does),
//! `torchaudio` chooses the torchaudio build a ROCm 10 or newer install pins,
//! `next_hint` words what a failed ROCm 10 resolution suggests next, and the
//! rest of `therock` reads the names below.

mod next_hint;
mod tarball;
mod torchaudio;
mod uv_access;
mod wheel;

pub(super) use self::tarball::{resolve_tarball_artifact, resolve_tarball_artifact_with_timeout};
pub(super) use self::torchaudio::{TorchaudioPin, unverified_torchaudio_note};
#[cfg(test)]
pub(super) use self::wheel::select_rocm_version;
pub(super) use self::wheel::{
    AggregateDeviceTarget, PipRuntimePlan, UnresolvedPipRuntime,
    canonical_auto_select_needs_next_python, preview_pip_runtime, resolve_pip_runtime,
    resolve_pip_runtime_with_timeout, wheel_runtime_composition,
};
