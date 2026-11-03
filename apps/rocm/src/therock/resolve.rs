// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! How a request is turned into the artifact an install fetches. Only module
//! declarations and re-exports live here, no logic: `tarball` resolves the SDK
//! tarball from the catalog's index, and `therock.rs` reads the names below.

mod tarball;

pub(super) use self::tarball::{resolve_tarball_artifact, resolve_tarball_artifact_with_timeout};
