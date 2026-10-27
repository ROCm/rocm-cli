// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Thin wrapper around the `docker` CLI binary. No `bollard`/Docker API
//! dependency: shelling out mirrors the pattern `engines/lemonade/src/process.rs`
//! already uses for its own CLI (`run_lemonade_pull`/`run_lemonade_backend_install`).

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::Duration;

pub(crate) const DOCKER_BIN: &str = "docker";

/// Bound for the short-lived `docker` calls (`inspect`, `logs`, `rm`, `kill`),
/// so a wedged daemon cannot hang the supervisor or a stop request.
const CLI_TIMEOUT: Duration = Duration::from_secs(30);

fn docker_output(command: Command, timeout: Duration, label: &str) -> Result<Output> {
    rocm_core::process::run_with_timeout(command, timeout, label)
}

pub(crate) struct RunSpec<'a> {
    pub name: &'a str,
    pub image: &'a str,
    pub host: &'a str,
    pub port: u16,
    pub container_port: u16,
    /// Whether to pass GPU devices through at all.
    pub use_gpu: bool,
    /// Pins `HIP_VISIBLE_DEVICES`; empty leaves the container's default visibility.
    pub gpu_indices: &'a [u32],
    pub render_gid: Option<u32>,
    /// `KEY=VALUE` entries go through a 0600 `--env-file`, never argv. An empty
    /// value means "forward `KEY` from this process's environment" (`-e KEY`).
    pub env: &'a BTreeMap<String, String>,
}

/// Splits `env` into entries destined for the `--env-file` and bare names to
/// forward from the engine process's own environment.
fn split_env(env: &BTreeMap<String, String>) -> (Vec<(&str, &str)>, Vec<&str>) {
    let mut file_entries = Vec::new();
    let mut forwarded = Vec::new();
    for (key, value) in env {
        if value.is_empty() {
            forwarded.push(key.as_str());
        } else {
            file_entries.push((key.as_str(), value.as_str()));
        }
    }
    (file_entries, forwarded)
}

fn run_args(
    spec: &RunSpec<'_>,
    wsl: bool,
    env_file: Option<&Path>,
    forwarded: &[&str],
) -> Vec<String> {
    let mut args = vec![
        "run".to_owned(),
        "-d".to_owned(),
        "--name".to_owned(),
        spec.name.to_owned(),
        "-p".to_owned(),
        format!("{}:{}:{}", spec.host, spec.port, spec.container_port),
    ];
    if spec.use_gpu {
        if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(spec.gpu_indices) {
            args.push("-e".to_owned());
            args.push(format!("HIP_VISIBLE_DEVICES={csv}"));
        }
        args.extend(gpu_device_args(wsl, spec.render_gid));
    }
    if let Some(path) = env_file {
        args.push("--env-file".to_owned());
        args.push(path.display().to_string());
    }
    for key in forwarded {
        args.push("-e".to_owned());
        args.push((*key).to_owned());
    }
    args.push(spec.image.to_owned());
    args
}

fn write_env_file(entries: &[(&str, &str)]) -> Result<tempfile::NamedTempFile> {
    // `NamedTempFile` is created 0600 on unix and removed on drop.
    let mut file = tempfile::NamedTempFile::new().context("failed to create docker env file")?;
    for (key, value) in entries {
        if key.contains(['=', '\n']) || value.contains('\n') {
            bail!("engine env `{key}` cannot contain newlines or `=` in its name");
        }
        writeln!(file, "{key}={value}").context("failed to write docker env file")?;
    }
    file.flush().context("failed to write docker env file")?;
    Ok(file)
}

pub(crate) fn run(spec: &RunSpec<'_>) -> Result<()> {
    let (file_entries, forwarded) = split_env(spec.env);
    for key in &forwarded {
        if std::env::var_os(key).is_none() {
            bail!(
                "engine env `{key}` was passed without a value, but `{key}` is not set in this \
                 process's environment; export it before serving or pass `{key}=VALUE`"
            );
        }
    }
    let env_file = if file_entries.is_empty() {
        None
    } else {
        Some(write_env_file(&file_entries)?)
    };
    let args = run_args(
        spec,
        rocm_core::is_wsl_host(),
        env_file.as_ref().map(tempfile::NamedTempFile::path),
        &forwarded,
    );

    let mut command = Command::new(DOCKER_BIN);
    command.args(&args).stdin(Stdio::null());
    // `docker run -d` returns once the container is created, but a first run
    // can take a while on a busy daemon; bound it generously rather than not at all.
    let output =
        docker_output(command, Duration::from_mins(2), "docker run").with_context(|| {
            format!(
                "failed to run `{DOCKER_BIN} run` for container `{}`",
                spec.name
            )
        })?;
    if !output.status.success() {
        bail!(
            "`docker run --name {}` failed: {}",
            spec.name,
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

/// Returns `Ok(false)` when there was nothing running to stop (no such
/// container, or it already exited).
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
    let label = if force { "docker kill" } else { "docker stop" };
    let output = docker_output(command, grace + CLI_TIMEOUT, label)
        .with_context(|| format!("failed to run `{label}` for container `{name}`"))?;
    if !output.status.success() {
        let detail = rocm_core::command_failure_detail(&output);
        if is_nothing_to_stop(&detail) {
            return Ok(false);
        }
        bail!("failed to stop container `{name}`: {detail}");
    }
    Ok(true)
}

/// Docker words a missing object as `No such container` or, in newer
/// releases, `error: no such object`; match both regardless of case.
fn is_no_such(detail: &str) -> bool {
    detail.to_ascii_lowercase().contains("no such")
}

fn is_nothing_to_stop(detail: &str) -> bool {
    let detail = detail.to_ascii_lowercase();
    detail.contains("no such container") || detail.contains("is not running")
}

/// Force-removes the container (stopping it first if needed) so its name and
/// writable layer, including any downloaded weights, are released. A container
/// that is already gone is not an error.
pub(crate) fn remove(name: &str) -> Result<()> {
    let mut command = Command::new(DOCKER_BIN);
    command.arg("rm").arg("-f").arg(name);
    let output = docker_output(command, CLI_TIMEOUT, "docker rm")
        .with_context(|| format!("failed to run `docker rm` for container `{name}`"))?;
    if !output.status.success() {
        let detail = rocm_core::command_failure_detail(&output);
        if is_no_such(&detail) {
            return Ok(());
        }
        bail!("failed to remove container `{name}`: {detail}");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContainerState {
    Running,
    /// Exited (or created but never started) with this exit code.
    Exited(i32),
    /// No container by this name exists.
    Missing,
}

/// A daemon that is unreachable or erroring is an `Err`, distinct from a
/// container that is definitively gone.
pub(crate) fn container_state(name: &str) -> Result<ContainerState> {
    let mut command = Command::new(DOCKER_BIN);
    command
        .arg("inspect")
        .arg("--format")
        .arg("{{.State.Running}} {{.State.ExitCode}}")
        .arg(name);
    let output = docker_output(command, CLI_TIMEOUT, "docker inspect")
        .with_context(|| format!("failed to run `docker inspect` for container `{name}`"))?;
    if !output.status.success() {
        let detail = rocm_core::command_failure_detail(&output);
        if is_no_such(&detail) {
            return Ok(ContainerState::Missing);
        }
        bail!("`docker inspect {name}` failed: {detail}");
    }
    parse_container_state(&String::from_utf8_lossy(&output.stdout))
}

fn parse_container_state(text: &str) -> Result<ContainerState> {
    let mut parts = text.split_whitespace();
    match (parts.next(), parts.next().map(str::parse::<i32>)) {
        (Some("true"), _) => Ok(ContainerState::Running),
        (Some("false"), Some(Ok(code))) => Ok(ContainerState::Exited(code)),
        _ => bail!("unexpected `docker inspect` output: {:?}", text.trim()),
    }
}

pub(crate) fn logs(name: &str, tail: usize) -> Result<Vec<String>> {
    let mut command = Command::new(DOCKER_BIN);
    command
        .arg("logs")
        .arg("--tail")
        .arg(tail.to_string())
        .arg(name);
    let output = docker_output(command, CLI_TIMEOUT, "docker logs")
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

/// The image's `Config` block (`Env`, `Labels`) from the local daemon, once the
/// image has been pulled. Works for any registry, unlike the Docker Hub lookup.
pub(crate) fn image_config(image: &str) -> Result<Option<serde_json::Value>> {
    let mut command = Command::new(DOCKER_BIN);
    command
        .arg("image")
        .arg("inspect")
        .arg("--format")
        .arg("{{json .Config}}")
        .arg(image);
    let output = docker_output(command, CLI_TIMEOUT, "docker image inspect")
        .with_context(|| format!("failed to run `docker image inspect {image}`"))?;
    if !output.status.success() {
        return Ok(None);
    }
    Ok(serde_json::from_slice(&output.stdout).ok())
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
    use super::{
        ContainerState, RunSpec, gpu_device_args, is_no_such, is_nothing_to_stop,
        parse_container_state, parse_getent_gid, run_args, split_env,
    };
    use std::collections::BTreeMap;
    use std::path::Path;

    fn spec<'a>(gpu_indices: &'a [u32], env: &'a BTreeMap<String, String>) -> RunSpec<'a> {
        RunSpec {
            name: "rocm-aim-svc",
            image: "org/aim-foo:1",
            host: "127.0.0.1",
            port: 8080,
            container_port: 8000,
            gpu_indices,
            use_gpu: !gpu_indices.is_empty(),
            render_gid: Some(993),
            env,
        }
    }

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
    fn run_args_without_gpu_or_env_is_just_name_port_and_image() {
        let env = BTreeMap::new();
        assert_eq!(
            run_args(&spec(&[], &env), false, None, &[]),
            vec![
                "run",
                "-d",
                "--name",
                "rocm-aim-svc",
                "-p",
                "127.0.0.1:8080:8000",
                "org/aim-foo:1"
            ]
        );
    }

    #[test]
    fn run_args_with_gpu_sets_visible_devices_and_device_flags_before_the_image() {
        let env = BTreeMap::new();
        let args = run_args(&spec(&[1, 3], &env), false, None, &[]);
        assert_eq!(args.last().map(String::as_str), Some("org/aim-foo:1"));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["-e", "HIP_VISIBLE_DEVICES=1,3"])
        );
        assert!(args.contains(&"--device=/dev/kfd".to_owned()));
        assert!(args.windows(2).any(|pair| pair == ["--group-add", "993"]));
    }

    #[test]
    fn run_args_with_gpu_and_no_pinned_indices_omits_visible_devices() {
        let env = BTreeMap::new();
        let mut gpu_spec = spec(&[], &env);
        gpu_spec.use_gpu = true;
        let args = run_args(&gpu_spec, false, None, &[]);
        assert!(
            !args
                .iter()
                .any(|arg| arg.starts_with("HIP_VISIBLE_DEVICES"))
        );
        assert!(args.contains(&"--device=/dev/kfd".to_owned()));
    }

    #[test]
    fn run_args_never_puts_env_values_on_the_command_line() {
        let env = BTreeMap::from([
            ("AIM_PROFILE_ID".to_owned(), "secret-profile".to_owned()),
            ("HF_TOKEN".to_owned(), String::new()),
        ]);
        let (file_entries, forwarded) = split_env(&env);
        assert_eq!(file_entries, vec![("AIM_PROFILE_ID", "secret-profile")]);
        assert_eq!(forwarded, vec!["HF_TOKEN"]);

        let args = run_args(
            &spec(&[], &env),
            false,
            Some(Path::new("/tmp/env-file")),
            &forwarded,
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--env-file", "/tmp/env-file"])
        );
        assert!(args.windows(2).any(|pair| pair == ["-e", "HF_TOKEN"]));
        assert!(!args.iter().any(|arg| arg.contains("secret-profile")));
        assert_eq!(args.last().map(String::as_str), Some("org/aim-foo:1"));
    }

    #[test]
    fn parse_container_state_reads_running_and_exit_code() {
        assert_eq!(
            parse_container_state("true 0\n").unwrap(),
            ContainerState::Running
        );
        assert_eq!(
            parse_container_state("false 137").unwrap(),
            ContainerState::Exited(137)
        );
        assert!(parse_container_state("").is_err());
        assert!(parse_container_state("false oops").is_err());
    }

    #[test]
    fn is_no_such_matches_every_docker_wording() {
        assert!(is_no_such("Error: No such object: x"));
        assert!(is_no_such("error: no such object: x"));
        assert!(is_no_such(
            "Error response from daemon: No such container: x"
        ));
        assert!(!is_no_such("Cannot connect to the Docker daemon"));
        assert!(is_nothing_to_stop("error: no such container: x"));
    }

    #[test]
    fn is_nothing_to_stop_covers_missing_and_already_exited_containers() {
        assert!(is_nothing_to_stop("Error: No such container: x"));
        assert!(is_nothing_to_stop(
            "Error response from daemon: cannot kill container: x: container x is not running"
        ));
        assert!(!is_nothing_to_stop("permission denied"));
    }
}

/// Exercises the `Command`-invoking functions against a fake `docker` on
/// `PATH`, mirroring `install.rs`'s subprocess pattern: mutating `PATH` for the
/// *current* process needs `unsafe` (denied workspace-wide), so each scenario
/// re-spawns this test binary filtered to just itself, with a hostile `PATH`
/// set via the safe `Command::env` on that child. The fake records every argv
/// (and a copy of any `--env-file`) so the parent can assert on them.
#[cfg(all(test, unix))]
pub(crate) mod subprocess_tests {
    use super::{ContainerState, RunSpec, container_state, pull, remove, run, stop};
    use rocm_core::fake_executable::write_fake_executable;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::time::Duration;

    const SENTINEL: &str = "ROCM_AIM_DOCKER_SUBPROCESS_TEST";

    /// One parameterized fake `docker`, driven entirely by inherited env vars,
    /// reused by every scenario below instead of planting N bespoke scripts.
    pub(crate) const FAKE_DOCKER_SCRIPT: &str = r#"#!/bin/sh
[ -n "$FAKE_DOCKER_ARGV_LOG" ] && echo "$*" >> "$FAKE_DOCKER_ARGV_LOG"
case "$1" in
  run)
    prev=""
    for arg in "$@"; do
      if [ "$prev" = "--env-file" ] && [ -n "$FAKE_DOCKER_ENV_COPY" ]; then
        cp "$arg" "$FAKE_DOCKER_ENV_COPY"
      fi
      prev="$arg"
    done
    [ -n "$FAKE_DOCKER_RUN_STDERR" ] && echo "$FAKE_DOCKER_RUN_STDERR" >&2
    exit "${FAKE_DOCKER_RUN_EXIT:-0}"
    ;;
  kill|stop)
    find "$FAKE_DOCKER_DIR/data" -name '*.json' -exec cat {} + > "$FAKE_DOCKER_DIR/state_at_stop" 2>/dev/null
    if [ -n "$FAKE_DOCKER_STOP_NO_SUCH_CONTAINER" ]; then
      echo "Error response from daemon: No such container: fake" >&2
      exit 1
    fi
    if [ -n "$FAKE_DOCKER_STOP_NOT_RUNNING" ]; then
      echo "Error response from daemon: container fake is not running" >&2
      exit 1
    fi
    [ -n "$FAKE_DOCKER_STOP_STDERR" ] && echo "$FAKE_DOCKER_STOP_STDERR" >&2
    exit "${FAKE_DOCKER_STOP_EXIT:-0}"
    ;;
  rm)
    if [ -n "$FAKE_DOCKER_RM_NO_SUCH_CONTAINER" ]; then
      echo "Error response from daemon: No such container: fake" >&2
      exit 1
    fi
    exit "${FAKE_DOCKER_RM_EXIT:-0}"
    ;;
  inspect)
    if [ -n "$FAKE_DOCKER_INSPECT_SEQUENCE" ]; then
      echo x >> "$FAKE_DOCKER_DIR/inspect.count"
      n=$(wc -l < "$FAKE_DOCKER_DIR/inspect.count" | tr -d ' ')
      entry=$(echo "$FAKE_DOCKER_INSPECT_SEQUENCE" | tr ',' '\n' | sed -n "${n}p")
      [ -z "$entry" ] && entry=$(echo "$FAKE_DOCKER_INSPECT_SEQUENCE" | tr ',' '\n' | tail -n 1)
      if [ "$entry" = "missing" ]; then
        echo "error: no such object: fake" >&2
        exit 1
      fi
      echo "$entry"
      exit 0
    fi
    if [ -n "$FAKE_DOCKER_INSPECT_MISSING" ]; then
      echo "error: no such object: fake" >&2
      exit 1
    fi
    if [ -n "$FAKE_DOCKER_INSPECT_ERROR" ]; then
      echo "Cannot connect to the Docker daemon" >&2
      exit 1
    fi
    echo "${FAKE_DOCKER_RUNNING:-false} ${FAKE_DOCKER_EXIT_CODE:-0}"
    exit 0
    ;;
  image)
    if [ -n "$FAKE_DOCKER_IMAGE_CONFIG" ]; then
      echo "$FAKE_DOCKER_IMAGE_CONFIG"
      exit 0
    fi
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

    pub(crate) fn in_subprocess() -> bool {
        std::env::var_os(SENTINEL).is_some()
    }

    /// Scratch dir holding the fake `docker` plus its argv/env-file recordings;
    /// removed on drop.
    pub(crate) struct Fixture {
        dir: PathBuf,
    }

    impl Fixture {
        pub(crate) fn argv(&self) -> Vec<String> {
            std::fs::read_to_string(self.dir.join("argv.log"))
                .unwrap_or_default()
                .lines()
                .map(ToOwned::to_owned)
                .collect()
        }

        /// The engine state files as they were when `docker kill`/`stop` ran.
        pub(crate) fn state_at_stop(&self) -> Option<String> {
            std::fs::read_to_string(self.dir.join("state_at_stop")).ok()
        }

        pub(crate) fn env_copy(&self) -> Option<String> {
            std::fs::read_to_string(self.dir.join("env.copy")).ok()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    /// Plants the fake `docker`, re-execs this test binary with just
    /// `test_path` selected and `extra_env` set, and asserts it succeeded.
    pub(crate) fn run_with_fake_docker(test_path: &str, extra_env: &[(&str, &str)]) -> Fixture {
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
        let fixture = Fixture { dir: dir.clone() };

        let mut entries = vec![fake_bin];
        entries.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));

        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", test_path, "--nocapture"])
            .env(SENTINEL, "1")
            .env("PATH", std::env::join_paths(entries).unwrap())
            .env("FAKE_DOCKER_ARGV_LOG", dir.join("argv.log"))
            .env("FAKE_DOCKER_ENV_COPY", dir.join("env.copy"))
            .env("FAKE_DOCKER_DIR", &dir)
            // Keep any state a child writes out of the user's real data dir.
            .env("ROCM_CLI_DATA_DIR", dir.join("data"))
            .stdin(Stdio::null());
        for (key, value) in extra_env {
            command.env(key, value);
        }
        let status = command.status().unwrap();
        assert!(status.success(), "helper subprocess failed for {test_path}");
        fixture
    }

    const fn plain_spec(env: &BTreeMap<String, String>) -> RunSpec<'_> {
        RunSpec {
            name: "fake-container",
            image: "fake-image",
            host: "127.0.0.1",
            port: 8080,
            container_port: 8000,
            use_gpu: false,
            gpu_indices: &[],
            render_gid: None,
            env,
        }
    }

    #[test]
    fn run_succeeds_against_fake_docker() {
        if in_subprocess() {
            run(&plain_spec(&BTreeMap::new())).expect("fake docker run should succeed");
            return;
        }
        let fixture = run_with_fake_docker(
            "docker::subprocess_tests::run_succeeds_against_fake_docker",
            &[],
        );
        assert_eq!(
            fixture.argv(),
            vec!["run -d --name fake-container -p 127.0.0.1:8080:8000 fake-image"]
        );
    }

    #[test]
    fn run_keeps_env_values_off_argv_and_in_a_private_env_file() {
        if in_subprocess() {
            let env = BTreeMap::from([
                ("AIM_PROFILE_ID".to_owned(), "profile-123".to_owned()),
                ("HF_TOKEN".to_owned(), String::new()),
            ]);
            run(&plain_spec(&env)).expect("fake docker run should succeed");
            return;
        }
        let fixture = run_with_fake_docker(
            "docker::subprocess_tests::run_keeps_env_values_off_argv_and_in_a_private_env_file",
            &[("HF_TOKEN", "hf_super_secret")],
        );
        let argv = fixture.argv().join("\n");
        assert!(argv.contains("--env-file"), "{argv}");
        assert!(argv.contains("-e HF_TOKEN"), "{argv}");
        assert!(!argv.contains("profile-123"), "{argv}");
        assert!(!argv.contains("hf_super_secret"), "{argv}");
        assert_eq!(
            fixture.env_copy().as_deref(),
            Some("AIM_PROFILE_ID=profile-123\n")
        );
    }

    #[test]
    fn run_refuses_a_forwarded_env_var_that_is_not_set() {
        if in_subprocess() {
            let env = BTreeMap::from([("ROCM_AIM_TEST_UNSET_VAR".to_owned(), String::new())]);
            let error = run(&plain_spec(&env)).expect_err("unset forwarded var must fail");
            assert!(error.to_string().contains("ROCM_AIM_TEST_UNSET_VAR"));
            return;
        }
        let fixture = run_with_fake_docker(
            "docker::subprocess_tests::run_refuses_a_forwarded_env_var_that_is_not_set",
            &[],
        );
        assert!(fixture.argv().is_empty(), "docker must not have been run");
    }

    #[test]
    fn run_bails_with_container_name_and_detail_on_failure() {
        if in_subprocess() {
            let error =
                run(&plain_spec(&BTreeMap::new())).expect_err("fake docker run should fail");
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
        let fixture = run_with_fake_docker(
            "docker::subprocess_tests::stop_force_kill_and_graceful_stop_both_succeed",
            &[],
        );
        assert_eq!(
            fixture.argv(),
            vec!["kill fake-container", "stop -t 1 fake-container"]
        );
    }

    #[test]
    fn stop_missing_or_already_exited_container_returns_false_not_error() {
        if in_subprocess() {
            let stopped = stop("fake-container", true, Duration::from_secs(1)).unwrap();
            assert!(!stopped);
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::stop_missing_or_already_exited_container_returns_false_not_error",
            &[("FAKE_DOCKER_STOP_NO_SUCH_CONTAINER", "1")],
        );
        run_with_fake_docker(
            "docker::subprocess_tests::stop_missing_or_already_exited_container_returns_false_not_error",
            &[("FAKE_DOCKER_STOP_NOT_RUNNING", "1")],
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
    fn remove_force_removes_and_tolerates_a_missing_container() {
        if in_subprocess() {
            remove("fake-container").expect("rm should succeed or tolerate a missing container");
            return;
        }
        let fixture = run_with_fake_docker(
            "docker::subprocess_tests::remove_force_removes_and_tolerates_a_missing_container",
            &[],
        );
        assert_eq!(fixture.argv(), vec!["rm -f fake-container"]);
        run_with_fake_docker(
            "docker::subprocess_tests::remove_force_removes_and_tolerates_a_missing_container",
            &[("FAKE_DOCKER_RM_NO_SUCH_CONTAINER", "1")],
        );
    }

    #[test]
    fn remove_surfaces_other_failures() {
        if in_subprocess() {
            let error = remove("fake-container").expect_err("should fail");
            assert!(error.to_string().contains("fake-container"), "{error}");
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::remove_surfaces_other_failures",
            &[("FAKE_DOCKER_RM_EXIT", "1")],
        );
    }

    #[test]
    fn container_state_distinguishes_running_exited_missing_and_daemon_errors() {
        if in_subprocess() {
            let state = container_state("fake-container");
            match std::env::var("FAKE_DOCKER_EXPECT").as_deref() {
                Ok("running") => assert_eq!(state.unwrap(), ContainerState::Running),
                Ok("exited") => assert_eq!(state.unwrap(), ContainerState::Exited(137)),
                Ok("missing") => assert_eq!(state.unwrap(), ContainerState::Missing),
                Ok("error") => assert!(state.is_err()),
                other => panic!("unexpected FAKE_DOCKER_EXPECT {other:?}"),
            }
            return;
        }
        let test = "docker::subprocess_tests::container_state_distinguishes_running_exited_missing_and_daemon_errors";
        run_with_fake_docker(
            test,
            &[
                ("FAKE_DOCKER_EXPECT", "running"),
                ("FAKE_DOCKER_RUNNING", "true"),
            ],
        );
        run_with_fake_docker(
            test,
            &[
                ("FAKE_DOCKER_EXPECT", "exited"),
                ("FAKE_DOCKER_EXIT_CODE", "137"),
            ],
        );
        run_with_fake_docker(
            test,
            &[
                ("FAKE_DOCKER_EXPECT", "missing"),
                ("FAKE_DOCKER_INSPECT_MISSING", "1"),
            ],
        );
        run_with_fake_docker(
            test,
            &[
                ("FAKE_DOCKER_EXPECT", "error"),
                ("FAKE_DOCKER_INSPECT_ERROR", "1"),
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
            assert!(super::image_present("fake-image").unwrap());
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::image_present_reports_true_when_fake_docker_reports_it",
            &[("FAKE_DOCKER_IMAGE_PRESENT_EXIT", "0")],
        );
    }

    #[test]
    fn image_config_parses_the_local_images_config_block() {
        if in_subprocess() {
            let config = super::image_config("fake-image").unwrap().expect("config");
            assert_eq!(config["Labels"]["k"], "v");
            return;
        }
        run_with_fake_docker(
            "docker::subprocess_tests::image_config_parses_the_local_images_config_block",
            &[(
                "FAKE_DOCKER_IMAGE_CONFIG",
                r#"{"Env":["A=b"],"Labels":{"k":"v"}}"#,
            )],
        );
    }
}
