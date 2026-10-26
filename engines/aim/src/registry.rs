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

/// Normalizes a user- or catalog-supplied image reference so the registry
/// metadata lookup and the real `docker pull`/`run` always resolve the exact
/// same image. A bare (org-less) name defaults to [`DEFAULT_ORG`], matching
/// this engine's own catalog convention — Docker's own bare-name default
/// (`library/`) would silently point at an unrelated official image instead.
pub(crate) fn normalize_image_ref(image_ref: &str) -> String {
    if image_ref.contains('/') {
        image_ref.to_owned()
    } else {
        format!("{DEFAULT_ORG}/{image_ref}")
    }
}

/// Splits a normalized (see [`normalize_image_ref`]) `org/repo[:tag|@digest]`
/// ref into `(repo, reference)`. Treats `@sha256:...` as a digest suffix, and
/// only looks for a tag's `:` after the last `/`, so a `host:port/repo` or
/// `repo@sha256:...` ref isn't mis-split by a naive first-`:` search.
fn split_image_ref(image_ref: &str) -> (String, String) {
    if let Some((name, digest)) = image_ref.split_once('@') {
        return (name.to_owned(), digest.to_owned());
    }
    let tag_sep = image_ref
        .rfind(':')
        .filter(|&i| i > image_ref.rfind('/').unwrap_or(0));
    match tag_sep {
        Some(i) => (image_ref[..i].to_owned(), image_ref[i + 1..].to_owned()),
        None => (image_ref.to_owned(), "latest".to_owned()),
    }
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
    let normalized = normalize_image_ref(image_ref);
    let (repo, reference) = split_image_ref(&normalized);
    let token = anonymous_pull_token(&repo)?;

    let manifest_url = format!("https://{REGISTRY_HOST}/v2/{repo}/manifests/{reference}");
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
    fn normalize_image_ref_defaults_org_for_bare_name() {
        assert_eq!(
            normalize_image_ref("aim-meta-llama-llama-3-2-1b-instruct:0.11.1"),
            "amdenterpriseai/aim-meta-llama-llama-3-2-1b-instruct:0.11.1"
        );
    }

    #[test]
    fn normalize_image_ref_preserves_explicit_org_or_registry() {
        assert_eq!(
            normalize_image_ref("amdenterpriseai/aim-foo:latest"),
            "amdenterpriseai/aim-foo:latest"
        );
        assert_eq!(
            normalize_image_ref("myregistry.example.com:5000/aim-foo:latest"),
            "myregistry.example.com:5000/aim-foo:latest"
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
    fn split_image_ref_handles_digest_refs() {
        assert_eq!(
            split_image_ref("amdenterpriseai/aim-foo@sha256:abcd1234"),
            (
                "amdenterpriseai/aim-foo".to_owned(),
                "sha256:abcd1234".to_owned()
            )
        );
    }

    #[test]
    fn split_image_ref_handles_registry_host_with_port() {
        // The first `:` here belongs to the registry host's port, not a tag
        // separator — a naive `split_once(':')` would mis-parse this as
        // repo="myregistry.example.com", tag="5000/aim-foo:v1".
        assert_eq!(
            split_image_ref("myregistry.example.com:5000/aim-foo:v1"),
            (
                "myregistry.example.com:5000/aim-foo".to_owned(),
                "v1".to_owned()
            )
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
