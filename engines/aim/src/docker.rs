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
        command.args(gpu_device_args(rocm_core::is_wsl_host(), render_gid));
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

/// GPU device/mount args for `docker run`, branched by host. WSL2 exposes the
/// GPU through the dxcore device model (`/dev/dxg`, with the host's dxcore
/// libs bind-mounted in since the container image doesn't ship them) — it has
/// no `/dev/kfd`, `/dev/dri`, or render-group gid, which are native-Linux
/// kernel concepts that don't apply there.
fn gpu_device_args(wsl: bool, render_gid: Option<u32>) -> Vec<String> {
    let mut args = vec!["--ipc=host".to_owned()];
    if wsl {
        args.push("--device=/dev/dxg".to_owned());
        args.push("-v".to_owned());
        args.push("/usr/lib/wsl/lib:/usr/lib/wsl/lib".to_owned());
    } else {
        args.push("--device=/dev/kfd".to_owned());
        args.push("--device=/dev/dri".to_owned());
        if let Some(gid) = render_gid {
            args.push("--group-add".to_owned());
            args.push(gid.to_string());
        }
    }
    args
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
    parse_getent_gid(&String::from_utf8_lossy(&output.stdout))
}

// `getent group render` prints `render:x:<gid>:<members>`.
fn parse_getent_gid(text: &str) -> Option<u32> {
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

/// Bounds `docker info` so a wedged daemon cannot hang this probe (and every
/// caller that gates on it, like `rocm model --verbose`/`rocm diagnose
/// --model`) for the OS's own TCP/socket-connect timeout.
const DAEMON_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

pub(crate) fn daemon_reachable() -> bool {
    let mut command = Command::new(DOCKER_BIN);
    command.arg("info");
    rocm_core::process::run_with_timeout(command, DAEMON_PROBE_TIMEOUT, "docker info")
        .is_ok_and(|output| output.status.success())
}

#[cfg(test)]
mod tests {
    use super::{extra_env_args, gpu_device_args, parse_getent_gid};
    use std::collections::BTreeMap;

    #[test]
    fn gpu_device_args_native_linux_uses_kfd_dri_and_render_gid() {
        assert_eq!(
            gpu_device_args(false, Some(993)),
            vec![
                "--ipc=host",
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--group-add",
                "993"
            ]
        );
    }

    #[test]
    fn gpu_device_args_wsl2_uses_dxg_device_and_wsl_lib_mount() {
        assert_eq!(
            gpu_device_args(true, Some(993)),
            vec![
                "--ipc=host",
                "--device=/dev/dxg",
                "-v",
                "/usr/lib/wsl/lib:/usr/lib/wsl/lib"
            ]
        );
    }

    #[test]
    fn render_group_gid_parses_getent_format() {
        assert_eq!(parse_getent_gid("render:x:993:"), Some(993));
        assert_eq!(parse_getent_gid("render:x:not-a-number:"), None);
        assert_eq!(parse_getent_gid(""), None);
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

/// Exercises the `Command`-invoking functions (`run`/`stop`/`inspect_*`/`pull`/
/// `image_present`) against a fake `docker` on `PATH`, mirroring `install.rs`'s
/// subprocess pattern: mutating `PATH` for the *current* process needs
/// `unsafe` (denied workspace-wide), so each scenario re-spawns this test
/// binary filtered to just itself, with a hostile `PATH` set via the safe
/// `Command::env` on that child.
#[cfg(all(test, unix))]
mod subprocess_tests {
    use super::{image_present, inspect_exit_code, inspect_running, pull, run, stop};
    use rocm_core::fake_executable::write_fake_executable;
    use std::collections::BTreeMap;
    use std::process::{Command, Stdio};
    use std::time::Duration;

    const SENTINEL: &str = "ROCM_AIM_DOCKER_SUBPROCESS_TEST";

    /// One parameterized fake `docker`, driven entirely by inherited env vars,
    /// reused by every scenario below instead of planting N bespoke scripts.
    const FAKE_DOCKER_SCRIPT: &str = r#"#!/bin/sh
case "$1" in
  run)
    [ -n "$FAKE_DOCKER_RUN_STDERR" ] && echo "$FAKE_DOCKER_RUN_STDERR" >&2
    exit "${FAKE_DOCKER_RUN_EXIT:-0}"
    ;;
  kill|stop)
    if [ -n "$FAKE_DOCKER_STOP_NO_SUCH_CONTAINER" ]; then
      echo "Error response from daemon: No such container: fake" >&2
      exit 1
    fi
    [ -n "$FAKE_DOCKER_STOP_STDERR" ] && echo "$FAKE_DOCKER_STOP_STDERR" >&2
    exit "${FAKE_DOCKER_STOP_EXIT:-0}"
    ;;
  inspect)
    case "$3" in
      *Running*) echo "${FAKE_DOCKER_RUNNING:-false}" ;;
      *ExitCode*) echo "${FAKE_DOCKER_EXIT_CODE:-0}" ;;
    esac
    exit 0
    ;;
  image)
    exit "${FAKE_DOCKER_IMAGE_PRESENT_EXIT:-1}"
    ;;
  pull)
    exit "${FAKE_DOCKER_PULL_EXIT:-0}"
    ;;
  logs)
    [ -n "$FAKE_DOCKER_LOGS" ] && echo "$FAKE_DOCKER_LOGS"
    exit 0
    ;;
  info)
    exit "${FAKE_DOCKER_INFO_EXIT:-0}"
    ;;
esac
exit 0
"#;

    fn in_subprocess() -> bool {
        std::env::var_os(SENTINEL).is_some()
    }

    /// Plants the fake `docker`, re-execs this test binary with just
    /// `test_path` selected and `extra_env` set, and asserts it succeeded.
    fn run_with_fake_docker(test_path: &str, extra_env: &[(&str, &str)]) {
        let dir = std::env::temp_dir().join(format!(
            "aim-docker-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fake_bin = dir.join("bin");
        std::fs::create_dir_all(&fake_bin).unwrap();
        write_fake_executable(&fake_bin.join("docker"), FAKE_DOCKER_SCRIPT).unwrap();

        let mut entries = vec![fake_bin];
        entries.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));

        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", test_path, "--nocapture"])
            .env(SENTINEL, "1")
            .env("PATH", std::env::join_paths(entries).unwrap())
            .stdin(Stdio::null());
        for (key, value) in extra_env {
            command.env(key, value);
        }
        let status = command.status().unwrap();
        assert!(status.success(), "helper subprocess failed for {test_path}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_succeeds_against_fake_docker() {
        if in_subprocess() {
            run(
                "fake-container",
                "fake-image",
                "127.0.0.1",
                8080,
                8000,
                &[],
                None,
                &BTreeMap::new(),
            )
            .expect("fake docker run should succeed");
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::run_succeeds_against_fake_docker",
            &[],
        );
    }

    #[test]
    fn run_bails_with_container_name_and_detail_on_failure() {
        if in_subprocess() {
            let error = run(
                "fake-container",
                "fake-image",
                "127.0.0.1",
                8080,
                8000,
                &[],
                None,
                &BTreeMap::new(),
            )
            .expect_err("fake docker run should fail");
            let message = error.to_string();
            assert!(message.contains("fake-container"), "{message}");
            assert!(message.contains("boom"), "{message}");
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::run_bails_with_container_name_and_detail_on_failure",
            &[
                ("FAKE_DOCKER_RUN_EXIT", "1"),
                ("FAKE_DOCKER_RUN_STDERR", "boom"),
            ],
        );
    }

    #[test]
    fn stop_force_kill_and_graceful_stop_both_succeed() {
        if in_subprocess() {
            assert!(stop("fake-container", true, Duration::from_secs(1)).unwrap());
            assert!(stop("fake-container", false, Duration::from_secs(1)).unwrap());
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::stop_force_kill_and_graceful_stop_both_succeed",
            &[],
        );
    }

    #[test]
    fn stop_missing_container_returns_false_not_error() {
        if in_subprocess() {
            let stopped = stop("fake-container", true, Duration::from_secs(1)).unwrap();
            assert!(!stopped);
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::stop_missing_container_returns_false_not_error",
            &[("FAKE_DOCKER_STOP_NO_SUCH_CONTAINER", "1")],
        );
    }

    #[test]
    fn stop_generic_failure_bails_with_detail() {
        if in_subprocess() {
            let error =
                stop("fake-container", true, Duration::from_secs(1)).expect_err("should fail");
            assert!(error.to_string().contains("disk full"), "{error}");
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::stop_generic_failure_bails_with_detail",
            &[
                ("FAKE_DOCKER_STOP_EXIT", "1"),
                ("FAKE_DOCKER_STOP_STDERR", "disk full"),
            ],
        );
    }

    #[test]
    fn inspect_running_and_exit_code_parse_fake_docker_output() {
        if in_subprocess() {
            assert_eq!(inspect_running("fake-container").unwrap(), Some(true));
            assert_eq!(inspect_exit_code("fake-container").unwrap(), Some(137));
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::inspect_running_and_exit_code_parse_fake_docker_output",
            &[
                ("FAKE_DOCKER_RUNNING", "true"),
                ("FAKE_DOCKER_EXIT_CODE", "137"),
            ],
        );
    }

    #[test]
    fn pull_bails_on_failure() {
        if in_subprocess() {
            let error = pull("fake-image").expect_err("fake docker pull should fail");
            assert!(error.to_string().contains("fake-image"), "{error}");
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::pull_bails_on_failure",
            &[("FAKE_DOCKER_PULL_EXIT", "1")],
        );
    }

    #[test]
    fn image_present_reports_true_when_fake_docker_reports_it() {
        if in_subprocess() {
            assert!(image_present("fake-image").unwrap());
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::image_present_reports_true_when_fake_docker_reports_it",
            &[("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0")],
        );
    }
}
