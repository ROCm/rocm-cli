// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Docker Hub v2 registry client for AIM image metadata (ROCMAI-553 finding):
//! the `amdenterpriseai` org's image config blob carries structured
//! `Env`/`Labels` fetchable via two plain HTTP calls, without pulling the image.

use anyhow::{Context, Result};
use serde_json::Value;
use std::time::Duration;

const REGISTRY_HOST: &str = "registry-1.docker.io";
const AUTH_HOST: &str = "auth.docker.io";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
pub(crate) const DEFAULT_ORG: &str = "amdenterpriseai";

#[derive(Debug, Default, Clone)]
pub(crate) struct AimImageMetadata {
    pub accelerator_family: Option<String>,
    pub accelerator_type: Option<String>,
    pub pytorch_rocm_arch: Vec<String>,
    pub canonical_name: Option<String>,
}

/// Splits `org/repo:tag` (or bare `repo:tag`, defaulting to [`DEFAULT_ORG`])
/// into `(repo, tag)` where `repo` includes the org prefix Docker Hub expects.
fn split_image_ref(image_ref: &str) -> (String, String) {
    let (name, tag) = image_ref.split_once(':').unwrap_or((image_ref, "latest"));
    let repo = if name.contains('/') {
        name.to_owned()
    } else {
        format!("{DEFAULT_ORG}/{name}")
    };
    (repo, tag.to_owned())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(DEFAULT_TIMEOUT)
        .timeout(DEFAULT_TIMEOUT)
        .build()
}

fn anonymous_pull_token(repo: &str) -> Result<String> {
    let url = format!(
        "https://{AUTH_HOST}/token?service=registry.docker.io&scope=repository:{repo}:pull"
    );
    let body: Value = agent()
        .get(&url)
        .set("User-Agent", "rocm-cli")
        .call()
        .with_context(|| format!("failed to fetch registry auth token for {repo}"))?
        .into_json()
        .with_context(|| format!("failed to parse registry auth token response for {repo}"))?;
    body.get("token")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .with_context(|| format!("registry auth response for {repo} had no token"))
}

pub(crate) fn fetch_image_metadata(image_ref: &str) -> Result<AimImageMetadata> {
    let (repo, tag) = split_image_ref(image_ref);
    let token = anonymous_pull_token(&repo)?;

    let manifest_url = format!("https://{REGISTRY_HOST}/v2/{repo}/manifests/{tag}");
    let manifest: Value = agent()
        .get(&manifest_url)
        .set("User-Agent", "rocm-cli")
        .set("Authorization", &format!("Bearer {token}"))
        .set(
            "Accept",
            "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json",
        )
        .call()
        .with_context(|| format!("failed to fetch manifest for {image_ref}"))?
        .into_json()
        .with_context(|| format!("failed to parse manifest for {image_ref}"))?;

    let config_digest = manifest
        .get("config")
        .and_then(|config| config.get("digest"))
        .and_then(Value::as_str)
        .with_context(|| format!("manifest for {image_ref} had no config digest"))?;

    let blob_url = format!("https://{REGISTRY_HOST}/v2/{repo}/blobs/{config_digest}");
    let config: Value = agent()
        .get(&blob_url)
        .set("User-Agent", "rocm-cli")
        .set("Authorization", &format!("Bearer {token}"))
        .call()
        .with_context(|| format!("failed to fetch config blob for {image_ref}"))?
        .into_json()
        .with_context(|| format!("failed to parse config blob for {image_ref}"))?;

    Ok(parse_config_blob(&config))
}

fn parse_config_blob(config: &Value) -> AimImageMetadata {
    let mut metadata = AimImageMetadata::default();

    let env_entries = config
        .pointer("/config/Env")
        .and_then(Value::as_array)
        .map(|entries| entries.iter().filter_map(Value::as_str).collect::<Vec<_>>())
        .unwrap_or_default();
    for entry in env_entries {
        if let Some((key, value)) = entry.split_once('=') {
            match key {
                "AIM_ACCELERATOR_FAMILY" => metadata.accelerator_family = Some(value.to_owned()),
                "AIM_ACCELERATOR_TYPE" => metadata.accelerator_type = Some(value.to_owned()),
                "PYTORCH_ROCM_ARCH" => {
                    metadata.pytorch_rocm_arch = value.split(';').map(ToOwned::to_owned).collect();
                }
                _ => {}
            }
        }
    }

    if let Some(labels) = config.pointer("/config/Labels").and_then(Value::as_object) {
        metadata.canonical_name = labels
            .get("com.amd.aim.model.canonicalName")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
    }

    metadata
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_image_ref_defaults_org_for_bare_repo() {
        assert_eq!(
            split_image_ref("aim-meta-llama-llama-3-2-1b-instruct:0.11.1"),
            (
                "amdenterpriseai/aim-meta-llama-llama-3-2-1b-instruct".to_owned(),
                "0.11.1".to_owned()
            )
        );
    }

    #[test]
    fn split_image_ref_preserves_explicit_org() {
        assert_eq!(
            split_image_ref("amdenterpriseai/aim-foo:latest"),
            ("amdenterpriseai/aim-foo".to_owned(), "latest".to_owned())
        );
    }

    #[test]
    fn split_image_ref_defaults_tag_to_latest() {
        assert_eq!(
            split_image_ref("amdenterpriseai/aim-foo"),
            ("amdenterpriseai/aim-foo".to_owned(), "latest".to_owned())
        );
    }

    #[test]
    fn parse_config_blob_extracts_known_fields() {
        let config = serde_json::json!({
            "config": {
                "Env": [
                    "PATH=/usr/bin",
                    "AIM_ACCELERATOR_FAMILY=radeon",
                    "AIM_ACCELERATOR_TYPE=gpu",
                    "PYTORCH_ROCM_ARCH=gfx1100;gfx1151",
                ],
                "Labels": {
                    "com.amd.aim.model.canonicalName": "Llama-3.2-1B-Instruct",
                },
            },
        });
        let metadata = parse_config_blob(&config);
        assert_eq!(metadata.accelerator_family.as_deref(), Some("radeon"));
        assert_eq!(metadata.accelerator_type.as_deref(), Some("gpu"));
        assert_eq!(metadata.pytorch_rocm_arch, vec!["gfx1100", "gfx1151"]);
        assert_eq!(
            metadata.canonical_name.as_deref(),
            Some("Llama-3.2-1B-Instruct")
        );
    }
}
