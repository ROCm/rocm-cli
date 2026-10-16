// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Much smaller than vLLM's/Lemonade's `install.rs`: there is no venv, wheel,
//! or torch-alignment logic. AIM images are self-contained containers, so
//! "install" for this engine means "Docker is present and reachable" — the
//! actual model image is pulled lazily per-launch (see `process::launch_service`),
//! since each `model_ref` names its own image.

use anyhow::{Result, bail};
use rocm_engine_protocol::InstallRequest;
use rocm_engine_protocol::InstallResponse;

pub(crate) fn install_response(request: InstallRequest) -> Result<InstallResponse> {
    rocm_core::require_nonempty(&request.runtime_id, "runtime_id")?;

    if !crate::docker::daemon_reachable() {
        bail!(
            "docker daemon is not reachable; install Docker and ensure it is running before using the AIM engine"
        );
    }

    Ok(InstallResponse {
        env_id: request.runtime_id,
        env_path: "docker".to_owned(),
        python_executable: "docker".to_owned(),
        runtime_kind: Some("docker".to_owned()),
        runtime_executable: Some("docker".to_owned()),
        managed_env: Some(false),
        installed_packages: Vec::new(),
        capabilities: crate::capabilities(),
        lock_hash: "docker".to_owned(),
        warnings: vec![
            "AIM model images are pulled on first launch, not during install".to_owned(),
        ],
    })
}
