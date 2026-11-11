// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `prefetch_artifact` sandbox tool's download policy and driver: whether a
//! recipe artifact may be fetched under the approved source policy, the
//! host-scoped Hugging Face URL checks that gate a token, and the streamed,
//! sha256-verified download that records the cache marker. [`crate::sandbox`]
//! resolves the artifact reference and calls in; nothing here calls back.

use crate::sandbox_tool::{SandboxToolArg, SandboxToolPolicy};
use anyhow::{Context, Result};
use rocm_core::atomic_write::write_file_atomically;
use rocm_core::{
    AppPaths, ModelRecipeArtifactRecord, model_artifact_cache_status, unix_time_millis,
};
use serde_json::Value;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Duration;

const ARTIFACT_PREFETCH_TIMEOUT: Duration = Duration::from_mins(10);

pub(crate) fn prefetch_artifact_value_with_policy(
    paths: &AppPaths,
    artifact_ref: &str,
    model: &str,
    artifact: ModelRecipeArtifactRecord,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    let cache = model_artifact_cache_status(paths, model, &artifact);
    if !policy.allow_artifact_download {
        return Ok(json!({
            "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
            "artifact_ref": artifact_ref,
            "model": model,
            "artifact": artifact,
            "cache": cache,
            "status": "source_policy_required",
            "mutating": false,
            "network_used": false,
            "message": "artifact prefetch requires an approved source policy before network access; no artifact bytes were downloaded",
        }));
    }

    if cache.marker_path.is_file() {
        return Ok(json!({
            "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
            "artifact_ref": artifact_ref,
            "model": model,
            "artifact": artifact,
            "cache": cache,
            "status": "cached",
            "mutating": false,
            "network_used": false,
            "message": "artifact cache marker already exists; no network request was made",
        }));
    }

    if let Some(message) = declared_source_policy_block_message(&artifact) {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            &message,
        ));
    }
    let source_policy_requires_huggingface_auth =
        artifact_declares_huggingface_authenticated_policy(&artifact);
    let artifact_kind = artifact.kind.to_ascii_lowercase();
    let huggingface_artifact = artifact_is_huggingface(&artifact);
    if artifact.gated.unwrap_or(false) && !huggingface_artifact {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "gated artifacts require a source-specific authentication policy before download",
        ));
    }
    if !matches!(artifact_kind.as_str(), "url" | "http" | "https" | "direct")
        && !huggingface_artifact
    {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "approved live prefetch currently supports direct HTTP(S) artifacts only",
        ));
    }
    if !artifact.uri.starts_with("https://") && !artifact.uri.starts_with("http://") {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            if huggingface_artifact {
                "authenticated Hugging Face prefetch requires an HTTP(S) artifact URI in the signed recipe metadata"
            } else {
                "approved live prefetch requires an HTTP(S) artifact URI"
            },
        ));
    }
    let mut request_headers = Vec::<(&str, String)>::new();
    let source_policy = if huggingface_artifact
        && (artifact.gated.unwrap_or(false)
            || policy.allow_huggingface_download
            || source_policy_requires_huggingface_auth)
    {
        if !policy.allow_huggingface_download {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "Hugging Face artifacts require --allow-huggingface-download before rocm-cli may use an authentication token",
            ));
        }
        if !artifact.uri.starts_with("https://") {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "rocm-cli will not send a Hugging Face token over plain HTTP; use an HTTPS Hugging Face artifact URI",
            ));
        }
        if !is_huggingface_url(&artifact.uri) {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "rocm-cli will not send a Hugging Face token to a non-Hugging Face URL",
            ));
        }
        let Some(token) = policy
            .huggingface_token
            .as_deref()
            .map(str::trim)
            .filter(|token| !token.is_empty())
        else {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "Hugging Face artifact prefetch needs ROCM_CLI_HUGGINGFACE_TOKEN, HF_TOKEN, or HUGGING_FACE_HUB_TOKEN",
            ));
        };
        request_headers.push(("Authorization", format!("Bearer {token}")));
        "huggingface_authenticated"
    } else {
        "explicit_allow_artifact_download"
    };
    let Some(expected_sha256) = artifact
        .sha256
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit()))
    else {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "approved live prefetch requires a valid sha256 in the recipe artifact metadata",
        ));
    };
    let Some(size_bytes) = artifact.size_bytes else {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "approved live prefetch requires size_bytes in the recipe artifact metadata",
        ));
    };
    let max_bytes = policy.artifact_max_bytes.unwrap_or(size_bytes);
    if size_bytes > max_bytes {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "artifact size exceeds the approved download byte limit",
        ));
    }

    // Streamed straight to its final path: model artifacts run to gigabytes, so
    // buffering one to hash it would hold the whole file in memory. The size and
    // digest from the recipe are enforced inside the download, which also gives
    // this path a free-space preflight and resume on a dropped connection.
    let artifact_path = artifact_bytes_path_for_marker(&cache.marker_path);
    let mut headers: Vec<(&str, &str)> = vec![("User-Agent", "rocm-cli")];
    headers.extend(
        request_headers
            .iter()
            .map(|(name, value)| (*name, value.as_str())),
    );
    let outcome = rocm_core::download_file_streaming(&rocm_core::DownloadRequest {
        url: &artifact.uri,
        destination: &artifact_path,
        timeout: ARTIFACT_PREFETCH_TIMEOUT,
        headers: &headers,
        max_bytes: Some(max_bytes),
        expected_len: Some(size_bytes),
        expected_sha256: Some(&expected_sha256),
    })
    .with_context(|| format!("failed to prefetch artifact `{artifact_ref}`"))?;
    let actual_sha256 = outcome.sha256;
    write_file_atomically(
        &cache.marker_path,
        &serde_json::to_vec_pretty(&json!({
            "artifact_ref": artifact_ref,
            "model": model,
            "artifact": artifact,
            "bytes_path": artifact_path,
            "size_bytes": size_bytes,
            "sha256": actual_sha256,
            "prefetched_at_unix_ms": unix_time_millis(),
            "source_policy": source_policy,
        }))
        .context("failed to serialize artifact cache marker")?,
    )?;
    let cache = model_artifact_cache_status(paths, model, &artifact);
    Ok(json!({
        "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
        "artifact_ref": artifact_ref,
        "model": model,
        "artifact": artifact,
        "cache": cache,
        "status": "prefetched",
        "mutating": true,
        "network_used": true,
        "bytes_path": artifact_path,
        "size_bytes": size_bytes,
        "sha256": actual_sha256,
        "source_policy": source_policy,
        "message": "artifact downloaded and verified with recipe sha256",
    }))
}

fn declared_source_policy_block_message(artifact: &ModelRecipeArtifactRecord) -> Option<String> {
    let source_policy = artifact.source_policy.as_ref()?;
    if !source_policy.required_hosts.is_empty() {
        let Some(host) = http_url_host(&artifact.uri) else {
            return Some(
                "recipe source policy declares required hosts but the artifact URI is not HTTP(S)"
                    .to_owned(),
            );
        };
        if !source_policy
            .required_hosts
            .iter()
            .any(|required| required.eq_ignore_ascii_case(&host))
        {
            return Some(format!(
                "artifact host `{host}` is not allowed by the recipe source policy"
            ));
        }
    }

    match source_policy.policy.as_str() {
        "direct_https_sha256" => {
            if artifact.uri.starts_with("https://") {
                None
            } else {
                Some(
                    "recipe source policy direct_https_sha256 requires an HTTPS artifact URI"
                        .to_owned(),
                )
            }
        }
        "huggingface_public" => {
            if artifact.gated.unwrap_or(false) {
                Some(
                    "recipe source policy marks this as public Hugging Face, but the artifact is gated"
                        .to_owned(),
                )
            } else {
                huggingface_policy_uri_block_message(artifact, "huggingface_public")
            }
        }
        "huggingface_authenticated" => {
            huggingface_policy_uri_block_message(artifact, "huggingface_authenticated")
        }
        "manual_only" => Some(
            "recipe source policy marks this artifact as manual-only; rocm-cli will not download it"
                .to_owned(),
        ),
        other => Some(format!(
            "recipe source policy `{other}` is not supported by this rocm-cli build"
        )),
    }
}

fn huggingface_policy_uri_block_message(
    artifact: &ModelRecipeArtifactRecord,
    policy: &str,
) -> Option<String> {
    if !artifact.uri.starts_with("https://") {
        return Some(format!(
            "recipe source policy {policy} requires an HTTPS Hugging Face artifact URI"
        ));
    }
    if !is_huggingface_url(&artifact.uri) {
        return Some(format!(
            "recipe source policy {policy} requires a Hugging Face artifact URI"
        ));
    }
    None
}

fn artifact_declares_huggingface_authenticated_policy(
    artifact: &ModelRecipeArtifactRecord,
) -> bool {
    artifact
        .source_policy
        .as_ref()
        .is_some_and(|policy| policy.policy == "huggingface_authenticated")
}

fn artifact_is_huggingface(artifact: &ModelRecipeArtifactRecord) -> bool {
    let kind = artifact.kind.to_ascii_lowercase();
    matches!(kind.as_str(), "huggingface" | "hf" | "hugging_face")
        || is_huggingface_url(&artifact.uri)
}

fn is_huggingface_url(url: &str) -> bool {
    http_url_host(url).is_some_and(|host| {
        host == "huggingface.co"
            || host.ends_with(".huggingface.co")
            || host == "hf.co"
            || host.ends_with(".hf.co")
    })
}

fn http_url_host(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .trim();
    if authority.is_empty() || authority.contains('@') {
        return None;
    }
    let host = authority
        .strip_prefix('[')
        .and_then(|value| value.split_once(']').map(|(host, _)| host))
        .unwrap_or_else(|| authority.split(':').next().unwrap_or_default())
        .trim()
        .trim_end_matches('.')
        .to_ascii_lowercase();
    (!host.is_empty()).then_some(host)
}

fn prefetch_blocked_value(
    artifact_ref: &str,
    model: &str,
    artifact: ModelRecipeArtifactRecord,
    cache: rocm_core::ModelArtifactCacheStatus,
    reason: &str,
) -> Value {
    json!({
        "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
        "artifact_ref": artifact_ref,
        "model": model,
        "artifact": artifact,
        "cache": cache,
        "status": "blocked",
        "mutating": false,
        "network_used": false,
        "message": reason,
    })
}

fn artifact_bytes_path_for_marker(marker_path: &Path) -> PathBuf {
    marker_path.with_extension("bin")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use sha2::Digest;
    use std::fs;
    use std::io::{Read, Write};
    use std::thread;

    /// Hex digest of a buffer. Production hashing happens incrementally inside the
    /// streaming download; this exists only so tests can state an expected digest.
    fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", sha2::Sha256::digest(bytes))
    }

    fn serve_one_http_response(bytes: Vec<u8>) -> Result<String> {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
        let addr = listener.local_addr()?;
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut request = [0_u8; 1024];
                let _ = stream.read(&mut request);
                let header = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    bytes.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&bytes);
            }
        });
        Ok(format!("http://{addr}/artifact.bin"))
    }

    #[test]
    fn sandbox_prefetch_reports_policy_required_without_network() {
        let (root, paths) = temp_app_paths("sandbox-prefetch-policy");
        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            ModelRecipeArtifactRecord {
                artifact_id: "hf-main".to_owned(),
                kind: "huggingface".to_owned(),
                uri: "Qwen/Test-1B".to_owned(),
                revision: Some("main".to_owned()),
                sha256: Some("a".repeat(64)),
                size_bytes: Some(1024),
                license: Some("apache-2.0".to_owned()),
                gated: Some(false),
                quantization: Some("bf16".to_owned()),
                engines: vec!["vllm".to_owned()],
                source_policy: None,
            },
            SandboxToolPolicy::default(),
        )
        .expect("policy value should render");
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("source_policy_required")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("cache")
                .and_then(|cache| cache.get("status"))
                .and_then(Value::as_str)
                .is_some_and(|status| status == "missing")
        );
    }

    #[test]
    fn sandbox_prefetch_cached_marker_skips_network() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-cached");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/should-not-be-fetched.bin".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };
        let cache = model_artifact_cache_status(&paths, "Qwen/Test-1B", &artifact);
        write_file_atomically(&cache.marker_path, br#"{"cached":true}"#)?;

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("cached"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_artifact_larger_than_approved_limit() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-byte-limit");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/too-large.bin".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(2048),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("byte limit"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_non_direct_non_huggingface_source() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-non-direct");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "torrent-bin".to_owned(),
            kind: "torrent".to_owned(),
            uri: "https://example.invalid/artifact.torrent".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#torrent-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("direct HTTP(S)"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_gated_huggingface_without_source_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-policy");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("--allow-huggingface-download"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_huggingface_source_policy_without_token() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-token");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                allow_huggingface_download: true,
                huggingface_token: None,
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("ROCM_CLI_HUGGINGFACE_TOKEN"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_respects_manual_only_source_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-manual-policy");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "manual-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/manual.bin".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: Some(rocm_core::ModelRecipeArtifactSourcePolicyRecord {
                policy: "manual_only".to_owned(),
                required_hosts: Vec::new(),
                notes: vec!["requires manual license review".to_owned()],
            }),
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#manual-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("manual-only"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_respects_declared_huggingface_auth_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-declared-auth");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: Some(rocm_core::ModelRecipeArtifactSourcePolicyRecord {
                policy: "huggingface_authenticated".to_owned(),
                required_hosts: vec!["huggingface.co".to_owned()],
                notes: Vec::new(),
            }),
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("--allow-huggingface-download"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_never_sends_huggingface_token_to_non_huggingface_url() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-host");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://example.invalid/Qwen/Test-1B/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                allow_huggingface_download: true,
                huggingface_token: Some("hf_secret_should_not_leak".to_owned()),
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("non-Hugging Face URL"))
        );
        assert!(
            !serde_json::to_string(&value)?.contains("hf_secret_should_not_leak"),
            "prefetch report must not include authentication tokens"
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_never_sends_huggingface_token_over_plain_http() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-https");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "http://huggingface.co/Qwen/Test-1B/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                allow_huggingface_download: true,
                huggingface_token: Some("hf_secret_should_not_leak".to_owned()),
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("plain HTTP"))
        );
        assert!(
            !serde_json::to_string(&value)?.contains("hf_secret_should_not_leak"),
            "prefetch report must not include authentication tokens"
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_downloads_direct_artifact_when_approved() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-download");
        let bytes = b"tiny verified artifact".to_vec();
        let uri = serve_one_http_response(bytes.clone())?;
        let sha256 = sha256_hex(&bytes);
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri,
            revision: None,
            sha256: Some(sha256.clone()),
            size_bytes: Some(bytes.len() as u64),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        let bytes_path = value
            .get("bytes_path")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .expect("bytes_path should be present");
        let marker_path = value
            .get("cache")
            .and_then(|cache| cache.get("marker_path"))
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .expect("marker path should be present");

        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("prefetched")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(true));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            value.get("sha256").and_then(Value::as_str),
            Some(sha256.as_str())
        );
        assert_eq!(
            value.get("source_policy").and_then(Value::as_str),
            Some("explicit_allow_artifact_download")
        );
        assert_eq!(fs::read(&bytes_path)?, bytes);
        assert!(marker_path.is_file());
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_approved_artifact_without_sha256() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-no-sha");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/artifact.bin".to_owned(),
            revision: None,
            sha256: None,
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("sha256"))
        );
        Ok(())
    }

    #[test]
    fn huggingface_url_detection_is_host_scoped() {
        assert!(is_huggingface_url(
            "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors"
        ));
        assert!(is_huggingface_url(
            "https://cdn-lfs.huggingface.co/repos/example"
        ));
        assert!(is_huggingface_url("https://hf.co/Qwen/Test-1B"));
        assert!(!is_huggingface_url(
            "https://huggingface.co.evil.example/Qwen/Test-1B"
        ));
        assert!(!is_huggingface_url(
            "https://huggingface.co@evil.example/Qwen/Test-1B"
        ));
    }
}
