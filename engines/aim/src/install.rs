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

#[cfg(all(test, unix))]
mod tests {
    use super::install_response;
    use rocm_core::fake_executable::write_fake_executable;
    use rocm_engine_protocol::InstallRequest;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    const DOCKER_UNREACHABLE_TEST_DIR: &str = "ROCM_AIM_DOCKER_UNREACHABLE_TEST_DIR";

    /// Child half of [`install_response_bails_clearly_when_docker_is_unreachable`].
    ///
    /// Runs in its own process so the hostile `PATH` (a fake `docker` that
    /// always fails) can be set with `Command::env` — mutating `PATH`
    /// in-process would need `unsafe`, which the workspace denies.
    #[test]
    fn docker_unreachable_subprocess_helper() {
        if std::env::var_os(DOCKER_UNREACHABLE_TEST_DIR).is_none() {
            return;
        }
        let request = InstallRequest {
            runtime_id: "aim".to_owned(),
            python_version: None,
            reinstall: false,
            env_root: None,
        };
        let error = install_response(request).expect_err("docker is unreachable");
        assert_eq!(
            error.to_string(),
            "docker daemon is not reachable; install Docker and ensure it is running before using the AIM engine"
        );
    }

    #[test]
    fn install_response_bails_clearly_when_docker_is_unreachable() {
        let dir = std::env::temp_dir().join(format!(
            "aim-install-docker-unreachable-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fake_bin = dir.join("bin");
        std::fs::create_dir_all(&fake_bin).unwrap();
        write_fake_executable(&fake_bin.join("docker"), "#!/bin/sh\nexit 1\n").unwrap();

        let mut entries = vec![fake_bin];
        entries.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));

        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "install::tests::docker_unreachable_subprocess_helper",
                "--nocapture",
            ])
            .env(DOCKER_UNREACHABLE_TEST_DIR, PathBuf::from("1"))
            .env("PATH", std::env::join_paths(entries).unwrap())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "helper subprocess failed");

        std::fs::remove_dir_all(&dir).ok();
    }
}
