// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Thin wrapper around the `docker` CLI binary. No `bollard`/Docker API
//! dependency: shelling out mirrors the pattern `engines/lemonade/src/process.rs`
//! already uses for its own CLI (`run_lemonade_pull`/`run_lemonade_backend_install`).

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::process::{Command, Stdio};
use std::time::Duration;

pub(crate) const DOCKER_BIN: &str = "docker";

/// Builds `-e KEY=VALUE` args for engine-specific env overrides (e.g. AIM's
/// `AIM_PROFILE_ID`). Sorted (`BTreeMap`) for deterministic argv in tests/logs.
fn extra_env_args(extra_env: &BTreeMap<String, String>) -> Vec<String> {
    extra_env
        .iter()
        .flat_map(|(key, value)| ["-e".to_owned(), format!("{key}={value}")])
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    name: &str,
    image: &str,
    host: &str,
    port: u16,
    container_port: u16,
    gpu_indices: &[u32],
    render_gid: Option<u32>,
    extra_env: &BTreeMap<String, String>,
) -> Result<()> {
    let mut command = Command::new(DOCKER_BIN);
    command
        .arg("run")
        .arg("-d")
        .arg("--name")
        .arg(name)
        .arg("-p")
        .arg(format!("{host}:{port}:{container_port}"));

    if !gpu_indices.is_empty() {
        if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(gpu_indices) {
            command.arg("-e").arg(format!("HIP_VISIBLE_DEVICES={csv}"));
        }
        command
            .arg("--device=/dev/kfd")
            .arg("--device=/dev/dri")
            .arg("--ipc=host");
        if let Some(gid) = render_gid {
            command.arg("--group-add").arg(gid.to_string());
        }
    }

    command.args(extra_env_args(extra_env));

    command.arg(image);
    command.stdin(Stdio::null());

    let output = command
        .output()
        .with_context(|| format!("failed to run `{DOCKER_BIN} run` for container `{name}`"))?;
    if !output.status.success() {
        bail!(
            "`docker run --name {name}` failed: {}",
            rocm_core::command_failure_detail(&output)
        );
    }
    Ok(())
}

/// Resolves the host's `render` group gid dynamically (host-specific, so it
/// cannot be hardcoded). Returns `None` if the group is absent (e.g. non-Linux).
pub(crate) fn render_group_gid() -> Option<u32> {
    let output = Command::new("getent")
        .arg("group")
        .arg("render")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    // `getent group render` prints `render:x:<gid>:<members>`.
    text.trim().split(':').nth(2)?.parse().ok()
}

pub(crate) fn stop(name: &str, force: bool, grace: Duration) -> Result<bool> {
    let mut command = Command::new(DOCKER_BIN);
    if force {
        command.arg("kill").arg(name);
    } else {
        command
            .arg("stop")
            .arg("-t")
            .arg(grace.as_secs().to_string())
            .arg(name);
    }
    let output = command.stdin(Stdio::null()).output().with_context(|| {
        format!(
            "failed to run `docker {}` for container `{name}`",
            if force { "kill" } else { "stop" }
        )
    })?;
    if !output.status.success() {
        // A container that's already gone is not a stop failure.
        let detail = rocm_core::command_failure_detail(&output);
        if detail.contains("No such container") {
            return Ok(false);
        }
        bail!("failed to stop container `{name}`: {detail}");
    }
    Ok(true)
}

pub(crate) fn inspect_running(name: &str) -> Result<Option<bool>> {
    inspect_format(name, "{{.State.Running}}").map(|value| value.map(|text| text == "true"))
}

pub(crate) fn inspect_exit_code(name: &str) -> Result<Option<i32>> {
    inspect_format(name, "{{.State.ExitCode}}")
        .map(|value| value.and_then(|text| text.parse().ok()))
}

fn inspect_format(name: &str, format: &str) -> Result<Option<String>> {
    let output = Command::new(DOCKER_BIN)
        .arg("inspect")
        .arg("--format")
        .arg(format)
        .arg(name)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("failed to run `docker inspect` for container `{name}`"))?;
    if !output.status.success() {
        return Ok(None);
    }
    Ok(Some(
        String::from_utf8_lossy(&output.stdout).trim().to_owned(),
    ))
}

pub(crate) fn logs(name: &str, tail: usize) -> Result<Vec<String>> {
    let output = Command::new(DOCKER_BIN)
        .arg("logs")
        .arg("--tail")
        .arg(tail.to_string())
        .arg(name)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("failed to run `docker logs` for container `{name}`"))?;
    if !output.status.success() {
        bail!(
            "`docker logs {name}` failed: {}",
            rocm_core::command_failure_detail(&output)
        );
    }
    let mut lines: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(ToOwned::to_owned)
        .collect();
    lines.extend(
        String::from_utf8_lossy(&output.stderr)
            .lines()
            .map(ToOwned::to_owned),
    );
    Ok(lines)
}

pub(crate) fn image_present(image: &str) -> Result<bool> {
    let status = Command::new(DOCKER_BIN)
        .arg("image")
        .arg("inspect")
        .arg(image)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("failed to run `docker image inspect {image}`"))?;
    Ok(status.success())
}

pub(crate) fn pull(image: &str) -> Result<()> {
    let status = Command::new(DOCKER_BIN)
        .arg("pull")
        .arg(image)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("failed to run `docker pull {image}`"))?;
    if !status.success() {
        bail!("`docker pull {image}` failed with status {status}");
    }
    Ok(())
}

pub(crate) fn daemon_reachable() -> bool {
    Command::new(DOCKER_BIN)
        .arg("info")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(test)]
mod tests {
    use super::extra_env_args;
    use std::collections::BTreeMap;

    #[test]
    fn render_group_gid_parses_getent_format() {
        let line = "render:x:993:";
        let gid: Option<u32> = line.trim().split(':').nth(2).and_then(|s| s.parse().ok());
        assert_eq!(gid, Some(993));
    }

    #[test]
    fn extra_env_args_builds_sorted_flags() {
        assert!(extra_env_args(&BTreeMap::new()).is_empty());

        let mut extra_env = BTreeMap::new();
        extra_env.insert("AIM_PROFILE_ID".to_owned(), "vllm-foo".to_owned());
        extra_env.insert("ANOTHER".to_owned(), "x".to_owned());
        assert_eq!(
            extra_env_args(&extra_env),
            vec!["-e", "AIM_PROFILE_ID=vllm-foo", "-e", "ANOTHER=x"]
        );
    }
}
