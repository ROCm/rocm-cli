// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Build and render the deterministic documentation demos.
//!
//! Environment setup happens here, before VHS starts. VHS advances its tape on
//! a fixed clock and does not wait for hidden setup commands, so the tapes stay
//! pure command/key sequences and run against an already-ready mock service.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

use crate::paths::{binary_name, release_binary_dir, workspace_root};

const DEFAULT_MODEL: &str = "Qwen/Qwen3.5-4B";
const DEMOS: &[&str] = &["cli", "console"];
const READY_TIMEOUT: Duration = Duration::from_secs(10);

pub fn run(names: &[String], skip_build: bool) -> Result<()> {
    let root = workspace_root()?;
    let demos = select_demos(names)?;
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());

    if !skip_build {
        build_binary(&cargo, &root, "rocm", "rocm")?;
        build_binary(&cargo, &root, "e2e-cucumber", "rocm-demo-env")?;
    }

    let bin_dir = release_binary_dir(&root, None);
    require_file(&bin_dir.join(binary_name("rocm")))?;
    require_file(&bin_dir.join(binary_name("rocm-demo-env")))?;

    let demo_root = unique_demo_root();
    std::fs::create_dir_all(&demo_root)
        .with_context(|| format!("failed to create {}", demo_root.display()))?;

    let mock = DemoProcess::start(&bin_dir, &demo_root)?;
    let env = mock.wait_until_ready()?;
    let path = prepend_path(&bin_dir)?;

    std::fs::create_dir_all(root.join("docs/media")).context("failed to create docs/media")?;

    for demo in demos {
        let tape = root.join("docs/tapes").join(format!("{demo}.tape"));
        require_file(&tape)?;
        eprintln!("rendering {demo} demo");
        let status = Command::new("vhs")
            .arg(&tape)
            .current_dir(&root)
            .env("PATH", &path)
            .envs(&env)
            .status()
            .with_context(|| format!("failed to run VHS for {}", tape.display()))?;
        if !status.success() {
            bail!("VHS failed while rendering the {demo} demo");
        }
    }

    Ok(())
}

fn select_demos(names: &[String]) -> Result<Vec<&'static str>> {
    if names.is_empty() {
        return Ok(DEMOS.to_vec());
    }

    let mut selected = Vec::new();
    for name in names {
        let Some(&known) = DEMOS.iter().find(|candidate| **candidate == name) else {
            bail!(
                "unknown demo `{name}`; expected one or more of: {}",
                DEMOS.join(", ")
            );
        };
        if !selected.contains(&known) {
            selected.push(known);
        }
    }
    Ok(selected)
}

fn build_binary(cargo: &OsStr, root: &Path, package: &str, binary: &str) -> Result<()> {
    let status = Command::new(cargo)
        .args(["build", "--release", "-p", package, "--bin", binary])
        .current_dir(root)
        .status()
        .with_context(|| format!("failed to build {binary}"))?;
    if !status.success() {
        bail!("building {binary} failed");
    }
    Ok(())
}

struct DemoProcess {
    child: Child,
    root: PathBuf,
    readiness: mpsc::Receiver<Result<HashMap<String, String>, String>>,
}

impl DemoProcess {
    fn start(bin_dir: &Path, root: &Path) -> Result<Self> {
        let model = std::env::var("ROCM_DEMO_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let executable = bin_dir.join(binary_name("rocm-demo-env"));
        let mut child = Command::new(&executable)
            .arg("--root")
            .arg(root)
            .args(["--model", &model])
            .env(
                "ROCM_MOCK_CHAT_REPLY",
                "ROCm CLI can inspect your system, manage runtimes and engines, serve models, monitor inference, and chat with local models.",
            )
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to start {}", executable.display()))?;

        let stdout = child
            .stdout
            .take()
            .context("rocm-demo-env stdout was not captured")?;
        let (tx, readiness) = mpsc::channel();
        std::thread::spawn(move || {
            let result = read_environment(BufReader::new(stdout));
            let _ = tx.send(result);
        });

        Ok(Self {
            child,
            root: root.to_path_buf(),
            readiness,
        })
    }

    fn wait_until_ready(&self) -> Result<HashMap<String, String>> {
        match self.readiness.recv_timeout(READY_TIMEOUT) {
            Ok(Ok(env)) => Ok(env),
            Ok(Err(message)) => bail!("rocm-demo-env failed before readiness: {message}"),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                bail!("timed out waiting for rocm-demo-env readiness")
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("rocm-demo-env readiness channel closed unexpectedly")
            }
        }
    }
}

impl Drop for DemoProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn read_environment(reader: impl BufRead) -> Result<HashMap<String, String>, String> {
    let mut env = HashMap::new();
    for line in reader.lines() {
        let line = line.map_err(|error| error.to_string())?;
        if let Some(export) = line.strip_prefix("export ") {
            let Some((key, value)) = export.split_once('=') else {
                return Err(format!("malformed environment line: {line}"));
            };
            if !key.starts_with("ROCM_CLI_") {
                return Err(format!("unexpected environment variable: {key}"));
            }
            env.insert(key.to_string(), value.to_string());
        }
        if line.contains("rocm-demo-env ready on") {
            for required in [
                "ROCM_CLI_CONFIG_DIR",
                "ROCM_CLI_DATA_DIR",
                "ROCM_CLI_CACHE_DIR",
            ] {
                if !env.contains_key(required) {
                    return Err(format!("readiness marker arrived without {required}"));
                }
            }
            return Ok(env);
        }
    }
    Err("process exited without a readiness marker".to_string())
}

fn prepend_path(bin_dir: &Path) -> Result<OsString> {
    let mut paths = vec![bin_dir.to_path_buf()];
    if let Some(current) = std::env::var_os("PATH") {
        paths.extend(std::env::split_paths(&current));
    }
    std::env::join_paths(paths).context("failed to prepend the demo binary directory to PATH")
}

fn unique_demo_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    std::env::temp_dir().join(format!("rocm-cli-demo-{}-{nonce}", std::process::id()))
}

fn require_file(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!("required file not found: {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn empty_selection_means_all_demos() {
        assert_eq!(select_demos(&[]).unwrap(), vec!["cli", "console"]);
    }

    #[test]
    fn selection_rejects_unknown_demo() {
        let error = select_demos(&["unknown".to_string()]).unwrap_err();
        assert!(error.to_string().contains("unknown demo `unknown`"));
    }

    #[test]
    fn readiness_parser_requires_all_isolated_paths() {
        let input =
            b"export ROCM_CLI_CONFIG_DIR=/tmp/config\n# rocm-demo-env ready on 127.0.0.1:1\n";
        let error = read_environment(Cursor::new(input)).unwrap_err();
        assert!(error.contains("ROCM_CLI_DATA_DIR"));
    }

    #[test]
    fn readiness_parser_returns_isolated_environment() {
        let input = b"export ROCM_CLI_CONFIG_DIR=/tmp/config\nexport ROCM_CLI_DATA_DIR=/tmp/data\nexport ROCM_CLI_CACHE_DIR=/tmp/cache\n# rocm-demo-env ready on 127.0.0.1:1\n";
        let env = read_environment(Cursor::new(input)).unwrap();
        assert_eq!(env["ROCM_CLI_CONFIG_DIR"], "/tmp/config");
        assert_eq!(env["ROCM_CLI_DATA_DIR"], "/tmp/data");
        assert_eq!(env["ROCM_CLI_CACHE_DIR"], "/tmp/cache");
    }
}
