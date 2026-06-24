// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Docker service discovery via `bollard`.
//!
//! Field-extraction logic is vendored from instinct-dash `DockerService.ts`
//! (HIP_VISIBLE_DEVICES → gpu_ids, `--tensor-parallel-size`/`-tp`, etc.).
//! See `../../wiki/entities/dockerode.md`.
//!
//! The async inherent methods are the public API. The `ServiceDiscovery` trait
//! impl is sync-only and returns `Unsupported`: bollard requires a tokio
//! runtime, and we don't want to silently block-on inside a sync caller.

use std::collections::BTreeMap;
use std::time::Duration;

use bollard::Docker;
use bollard::container::{InspectContainerOptions, ListContainersOptions};
use bollard::secret::ContainerInspectResponse;
use rocm_dash_core::traits::{CollectorError, DiscoveredService, Result, ServiceDiscovery};
use tokio::time::timeout;
use tracing::{debug, warn};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const LIST_TIMEOUT: Duration = Duration::from_secs(5);
const INSPECT_TIMEOUT: Duration = Duration::from_secs(5);

const DEFAULT_IMAGE_PATTERNS: &[&str] = &["vllm/*", "rocm/vllm*"];
/// Fallback vLLM port for container discovery when Docker exposes no host-port
/// binding. The discovered binding is the authority; for managed services the
/// registry `ManagedServiceRecord.port` is authoritative.
const DEFAULT_VLLM_PORT: u16 = 8000;

#[derive(Debug)]
pub struct DockerDiscovery {
    docker: Option<Docker>,
    image_patterns: Vec<String>,
}

impl Default for DockerDiscovery {
    fn default() -> Self {
        Self {
            docker: None,
            image_patterns: DEFAULT_IMAGE_PATTERNS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        }
    }
}

impl DockerDiscovery {
    /// Build with optional comma-separated image patterns (e.g. `"vllm/*,rocm/vllm*"`).
    pub fn new(image_patterns: Option<String>) -> Self {
        let patterns: Vec<String> = match image_patterns {
            Some(s) if !s.trim().is_empty() => s
                .split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect(),
            _ => DEFAULT_IMAGE_PATTERNS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        };
        Self {
            docker: None,
            image_patterns: patterns,
        }
    }

    /// Returns `Some` only if the local Docker daemon is reachable.
    pub async fn detect(image_patterns: Option<String>) -> Option<Self> {
        let mut me = Self::new(image_patterns);
        match Docker::connect_with_local_defaults() {
            Ok(d) => match timeout(CONNECT_TIMEOUT, d.ping()).await {
                Ok(Ok(_)) => {
                    me.docker = Some(d);
                    Some(me)
                }
                Ok(Err(e)) => {
                    warn!(error = %e, "docker ping failed");
                    None
                }
                Err(_) => {
                    warn!("docker ping timed out");
                    None
                }
            },
            Err(e) => {
                warn!(error = %e, "docker connect failed");
                None
            }
        }
    }

    pub async fn discover_async(&self) -> Result<Vec<DiscoveredService>> {
        let docker = self
            .docker
            .as_ref()
            .ok_or_else(|| CollectorError::Unsupported("docker not connected".into()))?;

        let opts: ListContainersOptions<String> = ListContainersOptions {
            all: false,
            ..Default::default()
        };
        let summaries = timeout(LIST_TIMEOUT, docker.list_containers(Some(opts)))
            .await
            .map_err(|_| CollectorError::Transport("docker list_containers timed out".into()))?
            .map_err(|e| CollectorError::Transport(format!("docker list_containers: {e}")))?;

        let mut out = Vec::new();
        for c in summaries {
            let image = c.image.as_deref().unwrap_or("");
            if !matches_any_pattern(image, &self.image_patterns) {
                continue;
            }
            let Some(id) = c.id.as_deref() else { continue };
            let inspect = timeout(
                INSPECT_TIMEOUT,
                docker.inspect_container(id, None::<InspectContainerOptions>),
            )
            .await
            .map_err(|_| CollectorError::Transport(format!("inspect {id} timed out")))?
            .map_err(|e| CollectorError::Transport(format!("inspect {id}: {e}")))?;
            match parse_container(&inspect) {
                Ok(svc) => out.push(svc),
                Err(e) => debug!(container_id = id, error = %e, "skipping container"),
            }
        }
        Ok(out)
    }
}

impl ServiceDiscovery for DockerDiscovery {
    fn name(&self) -> &'static str {
        "docker"
    }

    fn discover(&self) -> Result<Vec<DiscoveredService>> {
        Err(CollectorError::Unsupported(
            "DockerDiscovery is async — call discover_async() from a tokio runtime".into(),
        ))
    }
}

// --- pure helpers, fully unit-testable ---------------------------------------

fn matches_any_pattern(image: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|p| matches_pattern(image, p))
}

/// Glob-ish match: only `*` is special. Anchored at both ends, but a trailing
/// `:tag` on the image is allowed.
fn matches_pattern(image: &str, pattern: &str) -> bool {
    let base = image.split(':').next().unwrap_or(image);
    let pat_base = pattern.split(':').next().unwrap_or(pattern);
    glob_match(base, pat_base)
}

fn glob_match(s: &str, pat: &str) -> bool {
    // Backtracking glob over `*`.
    let (s, pat) = (s.as_bytes(), pat.as_bytes());
    let (mut i, mut j) = (0usize, 0usize);
    let (mut star, mut s_at_star) = (None::<usize>, 0usize);
    while i < s.len() {
        if j < pat.len() && pat[j] == b'*' {
            star = Some(j);
            s_at_star = i;
            j += 1;
        } else if j < pat.len() && pat[j] == s[i] {
            i += 1;
            j += 1;
        } else if let Some(sj) = star {
            j = sj + 1;
            s_at_star += 1;
            i = s_at_star;
        } else {
            return false;
        }
    }
    while j < pat.len() && pat[j] == b'*' {
        j += 1;
    }
    j == pat.len()
}

pub(crate) fn parse_env(env: &[String]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for entry in env {
        if let Some(eq) = entry.find('=')
            && eq > 0
        {
            out.insert(entry[..eq].to_string(), entry[eq + 1..].to_string());
        }
    }
    out
}

pub(crate) fn extract_gpu_ids(env: &BTreeMap<String, String>) -> Vec<String> {
    if let Some(v) = env.get("HIP_VISIBLE_DEVICES")
        && v != "all"
        && !v.is_empty()
    {
        return split_csv(v);
    }
    for k in [
        "AMD_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ] {
        if let Some(v) = env.get(k)
            && !v.is_empty()
        {
            return split_csv(v);
        }
    }
    Vec::new()
}

fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub(crate) fn extract_tensor_parallel(cmd: &[String]) -> u32 {
    let pairs = [("--tensor-parallel-size", 1u32), ("-tp", 1)];
    for (flag, default) in pairs {
        if let Some(i) = cmd.iter().position(|a| a == flag)
            && let Some(v) = cmd.get(i + 1)
        {
            return v.parse().unwrap_or(default);
        }
    }
    1
}

pub(crate) fn extract_arg<'a>(cmd: &'a [String], flag: &str) -> Option<&'a str> {
    cmd.iter()
        .position(|a| a == flag)
        .and_then(|i| cmd.get(i + 1))
        .map(String::as_str)
}

/// Effective quantization for a vLLM launch command. Prefers the explicit
/// `--quantization` flag; falls back to `--dtype` when no `--quantization` was
/// passed (the closest signal vLLM exposes). Returns `None` when neither flag
/// is present.
pub(crate) fn extract_quantization(cmd: &[String]) -> Option<String> {
    extract_arg(cmd, "--quantization")
        .or_else(|| extract_arg(cmd, "--dtype"))
        .map(str::to_string)
}

pub(crate) fn extract_model_name(cmd: &[String], env: &BTreeMap<String, String>) -> String {
    extract_arg(cmd, "--model")
        .map(str::to_string)
        .or_else(|| env.get("MODEL_NAME").cloned())
        .unwrap_or_else(|| "unknown".into())
}

fn parse_container(inspect: &ContainerInspectResponse) -> Result<DiscoveredService> {
    let id = inspect
        .id
        .clone()
        .ok_or_else(|| CollectorError::Parse("container missing id".into()))?;
    let name = inspect
        .name
        .clone()
        .unwrap_or_default()
        .trim_start_matches('/')
        .to_string();
    let (env_list, cmd, pid) = inspect
        .config
        .as_ref()
        .map(|cfg| {
            (
                cfg.env.clone().unwrap_or_default(),
                cfg.cmd.clone().unwrap_or_default(),
                inspect.state.as_ref().and_then(|s| s.pid).unwrap_or(0) as u32,
            )
        })
        .unwrap_or_default();

    let env = parse_env(&env_list);
    let gpu_ids = extract_gpu_ids(&env);
    let tp = extract_tensor_parallel(&cmd);
    let dtype = extract_arg(&cmd, "--dtype").map(str::to_string);
    let quantization = extract_quantization(&cmd);
    let model = extract_model_name(&cmd, &env);
    let port = extract_first_host_port(inspect).unwrap_or(DEFAULT_VLLM_PORT);

    Ok(DiscoveredService {
        container_id: id,
        container_name: name,
        model_name: model,
        gpu_ids,
        port: Some(port),
        tensor_parallel_size: tp,
        dtype,
        quantization,
        launch_args: cmd,
        env_vars: env,
        pid,
        log_file: None,
    })
}

fn extract_first_host_port(inspect: &ContainerInspectResponse) -> Option<u16> {
    let ports = inspect.network_settings.as_ref()?.ports.as_ref()?;
    for bindings in ports.values().flatten() {
        for b in bindings {
            if let Some(host_port) = &b.host_port
                && let Ok(p) = host_port.parse()
            {
                return Some(p);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches_vllm_image() {
        assert!(matches_pattern("vllm/vllm-openai:v0.5.0", "vllm/*"));
        assert!(matches_pattern("rocm/vllm-rocm:latest", "rocm/vllm*"));
        assert!(!matches_pattern("nginx:1.27", "vllm/*"));
        assert!(!matches_pattern("ghcr.io/vllm/vllm:latest", "vllm/*"));
    }

    #[test]
    fn glob_handles_pure_wildcard() {
        assert!(glob_match("anything", "*"));
        assert!(glob_match("", "*"));
        assert!(glob_match("foo-bar-baz", "foo*baz"));
        assert!(!glob_match("foo-bar", "foo*baz"));
    }

    #[test]
    fn parse_env_splits_on_first_equals() {
        let env = parse_env(&[
            "FOO=bar".into(),
            "EQ=a=b=c".into(),
            "NO_EQ".into(),
            "=leading_eq".into(),
        ]);
        assert_eq!(env.get("FOO").map(String::as_str), Some("bar"));
        assert_eq!(env.get("EQ").map(String::as_str), Some("a=b=c"));
        assert!(!env.contains_key("NO_EQ"));
        assert!(!env.contains_key(""));
    }

    #[test]
    fn gpu_ids_prefer_hip_over_amd() {
        let mut env = BTreeMap::new();
        env.insert("HIP_VISIBLE_DEVICES".into(), "0,2".into());
        env.insert("AMD_VISIBLE_DEVICES".into(), "all".into());
        assert_eq!(extract_gpu_ids(&env), vec!["0", "2"]);
    }

    #[test]
    fn gpu_ids_fall_back_when_hip_is_all() {
        let mut env = BTreeMap::new();
        env.insert("HIP_VISIBLE_DEVICES".into(), "all".into());
        env.insert("ROCR_VISIBLE_DEVICES".into(), "1,3".into());
        assert_eq!(extract_gpu_ids(&env), vec!["1", "3"]);
    }

    #[test]
    fn gpu_ids_empty_when_no_env() {
        assert!(extract_gpu_ids(&BTreeMap::new()).is_empty());
    }

    #[test]
    fn tensor_parallel_parses_long_and_short_flags() {
        let cmd = vec!["--tensor-parallel-size".into(), "8".into()];
        assert_eq!(extract_tensor_parallel(&cmd), 8);
        let cmd = vec!["serve".into(), "-tp".into(), "4".into()];
        assert_eq!(extract_tensor_parallel(&cmd), 4);
        let cmd: Vec<String> = vec!["serve".into()];
        assert_eq!(extract_tensor_parallel(&cmd), 1);
    }

    #[test]
    fn model_name_falls_back_to_env_then_unknown() {
        let env = parse_env(&["MODEL_NAME=llama3-70b".into()]);
        assert_eq!(extract_model_name(&[], &env), "llama3-70b");
        assert_eq!(extract_model_name(&[], &BTreeMap::new()), "unknown");
        let cmd = vec!["--model".into(), "deepseek".into()];
        assert_eq!(extract_model_name(&cmd, &env), "deepseek");
    }

    #[test]
    fn quantization_prefers_explicit_flag_then_dtype_then_none() {
        // Both flags present → --quantization wins.
        let cmd = vec![
            "--dtype".into(),
            "bfloat16".into(),
            "--quantization".into(),
            "fp8".into(),
        ];
        assert_eq!(extract_quantization(&cmd).as_deref(), Some("fp8"));

        // Only --dtype present → falls back to dtype value.
        let cmd = vec!["--dtype".into(), "float16".into()];
        assert_eq!(extract_quantization(&cmd).as_deref(), Some("float16"));

        // Only --quantization present.
        let cmd = vec!["--quantization".into(), "awq".into()];
        assert_eq!(extract_quantization(&cmd).as_deref(), Some("awq"));

        // Neither flag → None.
        let cmd = vec!["serve".into(), "--model".into(), "x".into()];
        assert_eq!(extract_quantization(&cmd), None);
    }

    #[test]
    fn sync_trait_returns_unsupported() {
        let d = DockerDiscovery::default();
        assert!(matches!(d.discover(), Err(CollectorError::Unsupported(_))));
    }
}
